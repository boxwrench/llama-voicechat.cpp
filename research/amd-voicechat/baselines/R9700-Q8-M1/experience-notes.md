# Experience notes

The deterministic text responses are relevant and complete for the five ordinary cases. The noisy case matches the clean factual answer. The pause case preserves the full question. The long case respects the requested itinerary constraints.

VC06 exercises the function channel and uses the supplied tool result in its spoken answer. The emitted call is structurally recognizable but is not strict JSON because it omits one colon. Treat function calling as qualified until the interactive driver has a parsing policy.

All measured output WAVs are non-empty and deterministic. This automated pass did not perform a human listening evaluation, so intelligibility and audio-artifact judgments remain unchecked.

Record whether each generated answer is intelligible, relevant, complete, free of repeated loops, and free of obvious audio artifacts. Keep subjective experience notes separate from deterministic hashes and timing measurements.
