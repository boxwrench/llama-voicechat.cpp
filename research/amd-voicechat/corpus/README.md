# Fixed VoiceChat corpus

The baseline corpus contains six short, fixed English inputs:

| ID | Purpose | Transcript |
| --- | --- | --- |
| VC01 | short factual | What's the capital of France? |
| VC02 | conversational | Explain why the sky looks blue in simple terms. |
| VC03 | long input | A 20 to 30 second question with several details. |
| VC04 | background noise | A short question mixed with ordinary room noise. |
| VC05 | pause handling | A question containing a long internal pause. |
| VC06 | function channel | A request paired with the frozen tool-capable system prompt. |

WAV files are generated or recorded once, then frozen by SHA256 in the baseline corpus manifest. They are not regenerated during benchmark runs.

