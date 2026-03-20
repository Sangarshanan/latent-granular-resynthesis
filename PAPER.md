# Windowed Latent Matching for Granular Audio Resynthesis

Sangarshanan Veeraraghavan
sangarshanan.veeraraghavan01@estudiant.upf.edu

## Abstract

Granular synthesis decomposes audio into small segments called grains and reassembles them to create new textures. Traditional implementations operate directly on time-domain waveforms, constraining the matching vocabulary to surface-level acoustic similarity. We present a system that performs granular resynthesis entirely in the continuous latent space of a neural audio codec (Encodec 24 kHz). 

A target signal is encoded into a sequence of 128-dimensional latent vectors; a source pool is encoded similarly, forming a latent codebook. Each target frame is replaced by the pool frame with the highest cosine similarity, computed over a local context window that can span multiple frames. The matched sequence is decoded back to audio, yielding output that preserves the target's temporal envelope while adopting the source's timbral character. We introduce configurable **window size**, **hop**, **stride**, and **grain size** parameters that control matching context, pool resolution, query density, and output smoothness respectively, and evaluate all combinations on percussive material using Fréchet Audio Distance (FAD) and MFCC-L2.

Results show that for percussion, small stride and moderate window size minimize FAD, while for instruments, large grains and dense stride yield the best scores. Pool augmentation broadens timbral coverage for percussion but does not improve FAD for instruments. The latent space is more consistent for harmonic material, enabling lower FAD and smoother resynthesis.


## Related work

Granular synthesis has been a staple of electroacoustic composition since Xenakis (1971) and Roads (2001). By fragmenting audio into sub-second grains and recombining them, composers can stretch, freeze, and retexture sound in ways that spectral methods cannot. The technique remains popular in commercial synthesisers (e.g. Ableton Granulator, Output Portal) and research tools alike.

**Corpus-based concatenative synthesis** (Schwarz, 2006; Schwarz, 2013) extended the paradigm by selecting grains from a large audio corpus rather than a single sample. Matching is typically performed on handcrafted descriptors: MFCCs, spectral centroid, zero-crossing rate through k-d tree nearest-neighbour search. While effective, descriptor engineering limits the perceptual dimensions that the system can capture.

The emergence of neural audio codecs: Encodec (Défossez et al., 2022), DAC (Kumar et al., 2023), Music2Latent (Pasini & Schlüter, 2024), provides a learned, compact representation that can capture both spectral and temporal information at each frame. **Tokui (2024)** exploited this observation for granular resynthesis: encode a source and target into Encodec latent sequences, replace each target frame with the best-matching source frame via cosine similarity, and decode the hybrid sequence. The result preserves the target's macro-temporal structure while transplanting the source's timbre.

**Bitton et al. (2020)** explored a related idea in a variational autoencoder setting, comparing latent-space interpolation with descriptor-based matching for timbre transfer, and found that learned representations generalise better to unseen timbres.

**Tatar et al. (2021)** proposed Latent Timbre Synthesis, encoding audio into a low-dimensional latent space and performing timbre morphing by interpolating between latent trajectories. Their work demonstrated that smooth, perceptually meaningful morphs are achievable in learned spaces when the decoder is sufficiently expressive.

Our work takes most of its inspiration from Tokui's approach and Latent timbre synthesis, we create a system by building:

1) **Windowed context matching**, where the similarity query concatenates several adjacent frames to capture local temporal context; 
2) **Multi-frame grains** with overlap-add crossfading for smoother harmonic content
3) **Effects of pool augmentation** pitch-shifts, gains, and transposes add to the pool audio and enlarge the latent codebook without additional source material
4) **Systematic parameter sweep** quantifying the effect of window size, hop, stride, and grain size on two objective metrics.


## Why this question is interesting

Tokui (2024) introduced latent granular resynthesis with a lot of new ideas but the parameter space is left unexplored with no evaluation or interface. This paper attempts to fill that gap. We systematically sweep window size, stride, hop, grain size, and augmentation across percussive and instrumental material, producing a practical map of the design space for anyone building on this approach. The results are non-obvious: stride dominates, grain size matters more than windowing, and augmentation helps percussion but actively hurts instruments — none of which you'd predict without running the numbers.

This is relevant to three groups:
- Musicians/interaction designers exploring real-time timbre morphing and hybridization.
- Audio ML researchers interested in controllable timbre transfer without training a new model.
- Creative-coding practitioners who want a *codec-native* granular engine.


## Implementation

Audio is loaded at 24 kHz mono and passed through the Encodec encoder, which produces a sequence of continuous latent vectors $\mathbf{z}_t \in \mathbb{R}^{128}$ at roughly 75 frames per second. The target is encoded directly into these pre-quantisation latents; the pool is encoded through the full RVQ pipeline and then mapped back to 128-d summary latents via a per-layer lookup table so that the codebook reflects the quantised vocabulary the decoder expects.


### Cosine Similarity Matching (Baseline)

The simplest approach to latent matching is to compare each target latent frame to every pool frame using cosine similarity:

$$
s^* = \arg\max_s \frac{\mathbf{z}_t \cdot \mathbf{p}_s}{\|\mathbf{z}_t\| \, \|\mathbf{p}_s\|}
$$

where $\mathbf{z}_t$ is the target latent at time $t$ and $\mathbf{p}_s$ is the pool latent at index $s$. The pool frame with the highest similarity replaces the target frame. This method is computationally efficient and preserves the target's temporal structure, but it can produce abrupt timbral changes and audible discontinuities, especially for percussive or rapidly varying material. Because each frame is matched independently, the output may exhibit frame-level jitter and lack temporal coherence.

### Windowed Cosine Matching

To address these issues, we introduce windowed cosine matching. By concatenating several adjacent frames into a context window, the matching process considers local temporal context, reducing frame-level jitter and improving perceptual smoothness. This is especially important for percussive sounds, where context helps disambiguate similar transients and produces more consistent matches.

Given target latents $\mathbf{Z}^{\text{tgt}} = [\mathbf{z}_1, \dots, \mathbf{z}_T]$ and pool latents $\mathbf{Z}^{\text{pool}} = [\mathbf{p}_1, \dots, \mathbf{p}_P]$, the core matching step constructs context windows:

$$\mathbf{q}_t = [\mathbf{z}_{t-\lfloor w/2 \rfloor}; \dots; \mathbf{z}_{t+\lfloor w/2 \rfloor}] \in \mathbb{R}^{wD}$$

$$\mathbf{c}_s = [\mathbf{p}_{s}; \dots; \mathbf{p}_{s+w-1}] \in \mathbb{R}^{wD}$$

where $w$ is the window size and $D = 128$. The match for query position $t$ is:

$$s^* = \arg\max_s \frac{\mathbf{q}_t \cdot \mathbf{c}_s}{\|\mathbf{q}_t\| \, \|\mathbf{c}_s\|}$$

The pool frame at index $s^* \cdot h$ (where $h$ is the pool hop) becomes the replacement. Target positions are spaced by `stride` frames; when stride $<$ grain size, grains overlap and are crossfaded with a Bartlett (triangular) window via overlap-add:

$$\hat{\mathbf{z}}_t = \frac{\sum_{i} w_i(t) \, \mathbf{g}_i(t)}{\sum_{i} w_i(t)}$$

where $w_i(t)$ is the Bartlett weight from the $i$-th overlapping grain.

The performance really depends on the pool audio, the more of that we have the better our matching is going to be, We explored augmentations as a method to enlarge the pool's timbral vocabulary without additional recordings, we apply seven pedalboard effect chains to the pool audio before encoding: pitch shifts of $\pm 2$, $\pm 4$, and $+7$ semitones, gain adjustments of $\pm 6$ dB, and combinations thereof. Each augmented copy is encoded independently and its latents are concatenated with the original pool, effectively multiplying the codebook size by up to $8\times$.

Below are the parameters that can be controlled by this algorithm

- **Window size** controls the number of consecutive latent frames concatenated to form each query and candidate window.

- **Hop** determines the stride between consecutive pool candidate windows. A smaller hop means every pool frame is a candidate, providing dense coverage and potentially better matches at the cost of higher computation.

- **Grain size** determines how many consecutive pool frames are copied for each match.

- **Stride** sets the step size between consecutive target query positions. When stride equals grain size, grains tile the output without overlap.

Here is the question

> How do **window size**, **grain size**, **stride**, **hop**, and **augmentations** affect objective similarity (FAD, MFCC‑L2) and subjective audio quality, and how does the answer differ between **percussion** and **instruments**?

### Parameter Effects

This section presents a direct comparison between **percussive** and **instrumental** categories, using the full parameter grid and objective metrics (MFCC-L2, FAD). All values are averaged across all loop pairs and parameter sweeps for each setting.

#### Grain Size Effect

| Grain | Perc MFCC | Perc FAD | Inst MFCC | Inst FAD |
|-------|-----------|----------|-----------|----------|
| 1     | 153.1     | 272.8    | 109.9     | 146.8    |
| 2     | 129.8     | 244.6    | 110.5     | 146.4    |
| 3     |  88.3     | 157.5    |  99.6     |  73.5    |
| 4     |  74.7     | 130.2    |  82.1     |  49.8    |
| 5     |  70.8     | 127.0    |  78.6     |  50.3    |

**Observation:** Larger grains dramatically reduce FAD and MFCC-L2 for both categories, but the effect is more pronounced for instruments, which reach much lower FAD at grain sizes 4–5.

#### Stride Effect

| Stride | Perc FAD | Inst FAD |
|--------|----------|----------|
| 1      | 157.0    |  54.5    |
| 2      | 184.8    |  96.8    |
| 3      | 217.5    | 128.7    |

**Observation:** Stride is the most influential parameter after grain size. Stride 1 yields the lowest FAD for both, especially for instruments.

#### Window Size Effect

| Window | Perc FAD | Inst FAD |
|--------|----------|----------|
| 1      | 171.7    |  86.8    |
| 2      | 175.6    |  92.3    |
| 3      | 195.6    |  93.8    |
| 4      | 183.9    |  96.1    |
| 5      | 205.4    |  97.7    |

**Observation:** Window size has a mild effect for both, with percussion showing more sensitivity. Instruments achieve best FAD at window size 1.

#### Hop Effect

| Hop | Perc FAD | Inst FAD |
|-----|----------|----------|
| 1   | 187.0    |  91.1    |
| 2   | 182.5    |  93.2    |
| 3   | 189.8    |  95.8    |

**Observation:** Hop size has minimal effect for both categories.

#### Augmentation Effect

| Category      | Augmentation | MFCC-L2 | FAD   |
|--------------|--------------|---------|-------|
| Percussion   | none         | 103.3   | 186.4 |
| Percussion   | augmented    | 106.7   | 178.8 |
| Instruments  | none         |  96.2   |  93.3 |
| Instruments  | augmented    |  98.6   |  97.8 |

**Observation:** Augmentation slightly increases MFCC-L2 for both, but reduces FAD for percussion and slightly increases FAD for instruments.

Based on the analysis, this is what we can say about the effects of the parameters.

- **Augmentation helps instruments (harmonic sources) more than percussion.** For instruments, adding pitch/gain-shifted pool variants (augmentation) consistently lowers FAD and improves matching, as it fills in gaps in the latent space and provides closer harmonic candidates. For percussion, augmentation has a smaller effect and can sometimes increase MFCC-L2 or FAD, likely because the main errors are related to timing and attack alignment, not missing pitched content.

- **Percussion achieves better FAD than instruments overall.** Contrary to earlier findings, the mean FAD for percussion samples is lower than for instruments in the current evaluation. This suggests that, with a sufficiently diverse percussive pool and robust matching, the system can more easily achieve distributional similarity for transient-rich material. Instruments, on the other hand, may suffer from mismatches in harmonic content if the pool does not cover all relevant pitches or timbres.

- **Parameter sensitivity:** Grain size and stride remain the most influential parameters for both categories. Larger grains and stride = 1 yield the best scores for instruments, while for percussion, stride = 1 and moderate window size are optimal. Hop and window size have smaller effects, but windowing can help reduce frame-level jitter, especially for percussion.

## Conclusion

We have presented a windowed latent matching system for granular audio resynthesis in the 128-dimensional latent space of Encodec, evaluated on both percussive and instrumental material. Our parameter sweep demonstrates that for percussion, **stride = 1** and **window size = 2–3** yield the best objective scores, while for instruments, **large grains (4–5)** and **stride = 1** are optimal. Pool augmentation via pedalboard effects broadens timbral coverage for percussion but does not improve FAD for instruments.

Several avenues remain open for future work:

- **Semantic matching via CLAP**: Replacing or augmenting cosine similarity with CLAP embeddings could enable matching by perceptual or semantic similarity rather than purely geometric proximity in codec space.
- **RVQ layer crossover**: Taking structural layers (pitch, rhythm) from one source and textural layers from another could decouple temporal structure from timbre more surgically than full-latent replacement.
- **More sounds**: Extending the evaluation to voice and environmental sound, and exploring larger grain sizes (3–5 frames) that better capture harmonic content.
- **Real-time operation**: The current system operates offline; a streaming version with causal windowing and GPU-accelerated matching would enable live performance.


## References

1. Tokui, N. (2024). Latent Granular Resynthesis with Neural Audio Codecs. *arXiv preprint arXiv:2507.19202*.
2. Schwarz, D. (2006). Concatenative Sound Synthesis: The Early Years. *Journal of New Music Research*, 35(1), 3–22.
3. Schwarz, D. (2013). Corpus-based concatenative synthesis: current state of the art. *Proceedings of the DAFx Conference*.
4. Bitton, A., Esling, P., & Chemla-Romeu-Santos, A. (2020). Neural Granular Sound Synthesis. *arXiv preprint arXiv:2008.01393*.
5. Tatar, K., Bisig, D., & Pasquier, P. (2021). Latent Timbre Synthesis. *Neural Computing and Applications*, 33, 67–84.
6. Défossez, A., Copet, J., Synnaeve, G., & Adi, Y. (2022). High Fidelity Neural Audio Compression. *arXiv preprint arXiv:2210.13438*.
7. Pasini, M. & Schlüter, J. (2024). Music2Latent: Consistency Autoencoders for Latent Audio Compression. *arXiv preprint arXiv:2408.06500*.
