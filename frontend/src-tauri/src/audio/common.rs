use crate::api::TranscriptSegment;
use anyhow::Result;
use log::{debug, info};
use once_cell::sync::Lazy;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};
use uuid::Uuid;

static ENGINE_LIFECYCLE_LOCK: Lazy<Arc<AsyncMutex<()>>> =
    Lazy::new(|| Arc::new(AsyncMutex::new(())));

pub(crate) async fn acquire_engine_lifecycle_lock() -> OwnedMutexGuard<()> {
    ENGINE_LIFECYCLE_LOCK.clone().lock_owned().await
}

/// Unload the transcription engine after a batch job (import or retranscription).
/// Skips unloading if a live recording is currently in progress, since recording
/// uses the same global engine instances.
pub(crate) async fn unload_engine_after_batch(use_parakeet: bool) {
    let _engine_lifecycle_guard = acquire_engine_lifecycle_lock().await;

    if crate::audio::recording_commands::is_recording().await {
        log::info!("Skipping model unload after batch: recording in progress");
        return;
    }

    if use_parakeet {
        use crate::parakeet_engine::commands::PARAKEET_ENGINE;
        let engine = {
            let guard = PARAKEET_ENGINE.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().cloned()
        };
        if let Some(e) = engine {
            e.unload_model().await;
        }
    } else {
        use crate::whisper_engine::commands::WHISPER_ENGINE;
        let engine = {
            let guard = WHISPER_ENGINE.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().cloned()
        };
        if let Some(e) = engine {
            e.unload_model().await;
        }
    }
}

/// Create transcript segments from transcription results.
/// Each tuple is (text, start_ms, end_ms) from VAD timestamps.
pub(crate) fn create_transcript_segments(transcripts: &[(String, f64, f64)]) -> Vec<TranscriptSegment> {
    transcripts
        .iter()
        .map(|(text, start_ms, end_ms)| {
            let start_seconds = start_ms / 1000.0;
            let end_seconds = end_ms / 1000.0;
            let duration = end_seconds - start_seconds;

            TranscriptSegment {
                id: format!("transcript-{}", Uuid::new_v4()),
                text: text.trim().to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
                audio_start_time: Some(start_seconds),
                audio_end_time: Some(end_seconds),
                duration: Some(duration),
            }
        })
        .collect()
}

/// Write transcripts.json to a meeting folder (atomic write with temp file)
pub(crate) fn write_transcripts_json(folder: &Path, segments: &[TranscriptSegment]) -> Result<()> {
    let transcript_path = folder.join("transcripts.json");
    let temp_path = folder.join(".transcripts.json.tmp");

    let json = serde_json::json!({
        "version": "1.0",
        "last_updated": chrono::Utc::now().to_rfc3339(),
        "total_segments": segments.len(),
        "segments": segments.iter().enumerate().map(|(i, s)| {
            serde_json::json!({
                "id": s.id,
                "text": s.text,
                "timestamp": s.timestamp,
                "audio_start_time": s.audio_start_time,
                "audio_end_time": s.audio_end_time,
                "duration": s.duration,
                "sequence_id": i
            })
        }).collect::<Vec<_>>()
    });

    let json_string = serde_json::to_string_pretty(&json)?;
    std::fs::write(&temp_path, &json_string)?;
    std::fs::rename(&temp_path, &transcript_path)?;

    info!(
        "Wrote transcripts.json with {} segments to {}",
        segments.len(),
        transcript_path.display()
    );
    Ok(())
}

/// Split a long speech segment at the lowest-energy (silence) point near the target size.
///
/// Scans for 100ms windows with minimal RMS energy within +/-3 seconds of each target
/// split point. If no clear silence is found, falls back to a 1-second overlap split
/// to avoid cutting words at boundaries.
pub(crate) fn split_segment_at_silence(
    segment: &crate::audio::vad::SpeechSegment,
    max_samples: usize,
) -> Vec<crate::audio::vad::SpeechSegment> {
    const SAMPLE_RATE: usize = 16000;
    // 100ms window for energy measurement (1600 samples at 16kHz)
    const ENERGY_WINDOW: usize = SAMPLE_RATE / 10;
    // Search +/-3 seconds around the target split point
    const SEARCH_RADIUS: usize = SAMPLE_RATE * 3;
    // RMS threshold below which we consider a window "silent"
    const SILENCE_RMS_THRESHOLD: f32 = 0.02;
    // Overlap to use when no silence boundary is found (1 second)
    const FALLBACK_OVERLAP: usize = SAMPLE_RATE;

    let total = segment.samples.len();
    if total <= max_samples {
        return vec![segment.clone()];
    }

    let ms_per_sample = (segment.end_timestamp_ms - segment.start_timestamp_ms)
        / segment.samples.len() as f64;
    let mut result = Vec::new();
    let mut pos = 0usize;

    while pos < total {
        let remaining = total - pos;
        if remaining <= max_samples {
            // Last chunk - take everything remaining
            let chunk_samples = segment.samples[pos..].to_vec();
            let chunk_start_ms = segment.start_timestamp_ms + (pos as f64 * ms_per_sample);
            let chunk_end_ms = segment.end_timestamp_ms;
            result.push(crate::audio::vad::SpeechSegment {
                samples: chunk_samples,
                start_timestamp_ms: chunk_start_ms,
                end_timestamp_ms: chunk_end_ms,
                confidence: segment.confidence,
            });
            break;
        }

        // Target split point
        let target = pos + max_samples;

        // Search window: [target - SEARCH_RADIUS, target + SEARCH_RADIUS]
        let search_start = target.saturating_sub(SEARCH_RADIUS).max(pos + SAMPLE_RATE);
        let search_end = (target + SEARCH_RADIUS).min(total.saturating_sub(ENERGY_WINDOW));

        // Find the lowest-energy 100ms window in the search range
        let mut best_split = target.min(total); // fallback: exact target
        let mut best_rms = f32::MAX;

        if search_start + ENERGY_WINDOW <= search_end {
            let mut idx = search_start;
            while idx + ENERGY_WINDOW <= search_end {
                let window = &segment.samples[idx..idx + ENERGY_WINDOW];
                let rms = (window.iter().map(|s| s * s).sum::<f32>() / ENERGY_WINDOW as f32).sqrt();
                if rms < best_rms {
                    best_rms = rms;
                    best_split = idx + ENERGY_WINDOW / 2; // split at center of quiet window
                }
                // Step by 10ms (160 samples) for efficiency
                idx += SAMPLE_RATE / 100;
            }
        }

        let split_at = best_split;
        if best_rms <= SILENCE_RMS_THRESHOLD {
            debug!(
                "Splitting at silence boundary: sample {} (RMS={:.4})",
                split_at, best_rms
            );
        } else {
            debug!(
                "No silence found near target (best RMS={:.4}), splitting with overlap at sample {}",
                best_rms, split_at
            );
        }

        // Determine the actual end of this chunk (with overlap if no silence)
        let chunk_end = if best_rms > SILENCE_RMS_THRESHOLD {
            (split_at + FALLBACK_OVERLAP).min(total)
        } else {
            split_at
        };

        let chunk_samples = segment.samples[pos..chunk_end].to_vec();
        let chunk_start_ms = segment.start_timestamp_ms + (pos as f64 * ms_per_sample);
        let chunk_end_ms = segment.start_timestamp_ms + (chunk_end as f64 * ms_per_sample);

        result.push(crate::audio::vad::SpeechSegment {
            samples: chunk_samples,
            start_timestamp_ms: chunk_start_ms,
            end_timestamp_ms: chunk_end_ms,
            confidence: segment.confidence,
        });

        // Advance position to where the current chunk actually ends
        // to avoid transcribing the overlap region twice
        pos = chunk_end;
    }

    result
}

/// Pack consecutive VAD speech segments up to `target_samples` per request.
///
/// Whisper is a fixed 30-second mel-window model, so a request much shorter than
/// the window gives the decoder little acoustic evidence per token and it falls
/// back on its language-model prior — which was trained on web subtitles, so the
/// failure mode is memorised boilerplate ("subscribe to the channel"), not
/// silence. Measured on real Arabic meeting audio: chunks under 2s hallucinated
/// at 19%, chunks of 5s or more at 1%. VAD should therefore choose segment
/// *boundaries* and never segment *sizes*.
///
/// `split_segment_at_silence` only caps the upper end. This fills the lower end
/// by concatenating consecutive segments, which also drops the silence between
/// them and so raises the speech-to-silence ratio — the other half of the
/// short-chunk problem (a ~1s VAD segment is ~60% padding).
///
/// Run this *after* splitting, so no input segment exceeds `target_samples`; an
/// over-long segment is passed through untouched rather than silently cut.
///
/// Boundaries are only ever original VAD boundaries, so no word is ever cut
/// mid-utterance. The packed segment spans `first.start_timestamp_ms` to
/// `last.end_timestamp_ms`; because inter-segment silence is dropped, that span
/// is longer than the packed audio, so transcript timestamps become accurate to
/// the packed group rather than to each individual utterance.
pub(crate) fn pack_segments_to_target(
    segments: &[crate::audio::vad::SpeechSegment],
    target_samples: usize,
) -> Vec<crate::audio::vad::SpeechSegment> {
    let mut packed: Vec<crate::audio::vad::SpeechSegment> = Vec::new();
    let mut current: Option<crate::audio::vad::SpeechSegment> = None;

    for segment in segments {
        current = match current {
            None => Some(segment.clone()),
            Some(accumulated) => {
                if accumulated.samples.len() + segment.samples.len() <= target_samples {
                    // Build a new segment rather than mutating the accumulator's
                    // identity: the Vec is local and owned, so extending it here
                    // is construction, not mutation of a caller's data.
                    let mut samples = accumulated.samples;
                    samples.extend_from_slice(&segment.samples);
                    Some(crate::audio::vad::SpeechSegment {
                        samples,
                        start_timestamp_ms: accumulated.start_timestamp_ms,
                        end_timestamp_ms: segment.end_timestamp_ms,
                        // A packed group is only as trustworthy as its weakest
                        // member. (Barely load-bearing: callers use the engine's
                        // confidence, not the VAD's, for the transcript.)
                        confidence: accumulated.confidence.min(segment.confidence),
                    })
                } else {
                    packed.push(accumulated);
                    Some(segment.clone())
                }
            }
        };
    }

    if let Some(accumulated) = current {
        packed.push(accumulated);
    }

    info!(
        "Packed {} VAD segments into {} requests (target {:.1}s each)",
        segments.len(),
        packed.len(),
        target_samples as f64 / 16000.0
    );

    packed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_lifecycle_lock_serializes_acquirers() {
        let guard = acquire_engine_lifecycle_lock().await;
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (acquired_tx, mut acquired_rx) = tokio::sync::oneshot::channel();
        let waiter = tokio::spawn(async {
            started_tx.send(()).unwrap();
            let _guard = acquire_engine_lifecycle_lock().await;
            acquired_tx.send(()).unwrap();
        });

        started_rx.await.unwrap();
        assert!(acquired_rx.try_recv().is_err());
        drop(guard);

        acquired_rx.await.unwrap();
        waiter.await.unwrap();
    }

    // ------------------------------------------------------------------
    // pack_segments_to_target
    // ------------------------------------------------------------------

    const SR: usize = 16000;
    const TARGET: usize = 25 * SR;

    /// A segment of `seconds` duration whose timestamps start at `start_s`.
    /// Timestamps intentionally leave a 1s gap when segments are laid out
    /// end-to-end by the callers below, mirroring real VAD output.
    fn segment(start_s: f64, seconds: f64) -> crate::audio::vad::SpeechSegment {
        crate::audio::vad::SpeechSegment {
            samples: vec![0.1; (seconds * SR as f64) as usize],
            start_timestamp_ms: start_s * 1000.0,
            end_timestamp_ms: (start_s + seconds) * 1000.0,
            confidence: 0.9,
        }
    }

    #[test]
    fn packing_merges_short_segments_up_to_target() {
        // Ten 3.5s segments -- the measured median of the broken live path.
        let segments: Vec<_> = (0..10).map(|i| segment(i as f64 * 4.5, 3.5)).collect();

        let packed = pack_segments_to_target(&segments, TARGET);

        // 3.5s * 7 = 24.5s fits; an eighth would reach 28s and must not.
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0].samples.len(), (3.5 * 7.0 * SR as f64) as usize);
        assert_eq!(packed[1].samples.len(), (3.5 * 3.0 * SR as f64) as usize);
    }

    #[test]
    fn packing_never_exceeds_target() {
        let segments: Vec<_> = (0..50).map(|i| segment(i as f64 * 2.0, 1.7)).collect();

        for group in pack_segments_to_target(&segments, TARGET) {
            assert!(
                group.samples.len() <= TARGET,
                "packed group of {} samples exceeds target {}",
                group.samples.len(),
                TARGET
            );
        }
    }

    #[test]
    fn packing_spans_first_start_to_last_end() {
        let segments = vec![segment(0.0, 3.0), segment(5.0, 3.0), segment(10.0, 3.0)];

        let packed = pack_segments_to_target(&segments, TARGET);

        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].start_timestamp_ms, 0.0);
        assert_eq!(packed[0].end_timestamp_ms, 13_000.0);
        // Inter-segment silence is dropped, so the audio is shorter than the span.
        assert_eq!(packed[0].samples.len(), (9.0 * SR as f64) as usize);
    }

    #[test]
    fn packing_preserves_total_audio_and_order() {
        let segments: Vec<_> = (0..13).map(|i| segment(i as f64 * 6.0, 5.0)).collect();
        let total: usize = segments.iter().map(|s| s.samples.len()).sum();

        let packed = pack_segments_to_target(&segments, TARGET);

        assert_eq!(packed.iter().map(|s| s.samples.len()).sum::<usize>(), total);
        assert!(
            packed
                .windows(2)
                .all(|w| w[0].end_timestamp_ms <= w[1].start_timestamp_ms),
            "packed groups must stay in chronological order"
        );
    }

    #[test]
    fn packing_passes_through_oversized_segment_untouched() {
        // Splitting runs first, so this should not occur -- but if it does, the
        // segment must survive rather than be silently cut.
        let long = segment(0.0, 40.0);

        let packed = pack_segments_to_target(&[long.clone()], TARGET);

        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].samples.len(), long.samples.len());
    }

    #[test]
    fn packing_handles_empty_input() {
        assert!(pack_segments_to_target(&[], TARGET).is_empty());
    }

    #[test]
    fn packing_takes_weakest_confidence_in_group() {
        let mut low = segment(5.0, 3.0);
        low.confidence = 0.42;
        let segments = vec![segment(0.0, 3.0), low];

        let packed = pack_segments_to_target(&segments, TARGET);

        assert_eq!(packed.len(), 1);
        assert!((packed[0].confidence - 0.42).abs() < f32::EPSILON);
    }
}
