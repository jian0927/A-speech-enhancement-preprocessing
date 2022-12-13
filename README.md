# A-speech-enhancement-preprocessing
preprocessing algorithm, emphasizing fundamental frequency and harmonics, DNN, SNMF
We embedded the EFF to emphasize the fundamental frequency as a supplementary treatment of speech enhancement algorithms. It can be used as a preprocessing or post-processing step in a speech enhancement system. This can make other algorithms perform better in the PESQ and WSS, but because emphasizing fundamental frequency will cause damage to some of the smaller structures, this leads to the less STOI scores. PESQ and WSS are auditory perception index, which directly feedback the effectiveness of people's speech perception. The preprocessing algorithm designed in this paper is based on this point, and the experimental data also proves that this algorithm has a significant improvement in perception. In fact, because the auditory system prefers to receive the intensity on the fundamental frequency and harmonics, emphasizing the fundamental frequency does not affect the auditory quality much.
 In addition, this improvement method starting from speech and auditory features has good portability, and can be further expanded in the direction of source separation on this basis. 
