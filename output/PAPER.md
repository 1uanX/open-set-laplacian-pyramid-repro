

[page 1]

A Radar Signal Open-Set
Deinterleaving Method Based
on Laplacian Pyramid
Reconstruction and Adversarial
Reciprocal Point Learning
WENBO LI
YANG-YANG DONG
, Member, IEEE
CHUNXI DONG
RONGHUA GUO
ZHIYUAN LI
Xidian University, Xi’an, China
Radar signal deinterleaving is a challenging task in complex elec-
tromagnetic environments. This work proposes a method for accu-
rately deinterleaving known and unknown radar radiation sources in
an open space using an open-set deinterleaving technique for radar
signals based on the combination of Laplacian pyramid reconstruc-
tion and reciprocal point adversarial deinterleaving (PR-RPAD). The
pulse description graph (PDG) representation of the intercepted pulse
description word is ﬁrst realized by applying a gray matrix symmetric
mapping approach based on sliding windows. Second, to improve
the visual characterization of the pulse description information of
various radar radiation sources, PDG edges and texture structures are
ampliﬁed and enhanced using the Otsu-threshold-based image ampli-
ﬁcation technique. Next, a Laplacian-pyramid-based multiresolution
feature reconstruction and fusion model is put forth, which uses a
jump connection and a multiplication gate to achieve tower reshaping
from low-resolution to high-resolution features. The idea of reciprocal
points is ﬁnally introduced to model the open space to decrease the
risk of closed-set deinterleaving through confrontation with known
radars. To improve the discriminatory nature of the model against
unknown radars and to ﬁnish the accurate deinterleaving of radar
radiation sources in the open space, an instantiated confrontation en-
hancement method is used to generate confusing samples for training.
In addition to having better performance in closed-set deinterleaving
Received 20 November 2024; revised 16 February 2025 and 4 April 2025;
accepted 17 April 2025. Date of publication 21 April 2025; date of current
version 13 October 2025.
DOI. No. 10.1109/TAES.2025.3563145
Refereeing of this contribution was handled by W. Al-Ashwal.
Authors’ address: Wenbo Li, Yang-Yang Dong, Chunxi Dong, Ronghua
Guo, and Zhiyuan Li are with the School of Electronic Engineering, Xi-
dian University, Xi’an 710071, China, E-mail: (lwb20220617@163.com;
dongyangyang2104@126.com; chxdong@mail.xidian.edu.cn; 22021110
341@stu.xidian.edu.cn; lizzyuan@163.com). (Corresponding author:
Yang-Yang Dong.)
0018-9251 © 2025 IEEE
than other deinterleaving techniques (sequential difference histogram
(SDIF), pulse repetition interval transform algorithm (PRI-Tran),
bidirectional long short-term memory (BLSTM), bidirectional gated
recurrent unit (BGRU), and dilated convolutional network (DCN)),
the PR-RPAD method also achieves deinterleaving in the open space,
which offers great versatility and potential for real-world use in
intricate electromagnetic environments.
I.
INTRODUCTION
One of the most important components of electronic
warfare is electronic support measures (ESM), a device
that measures the parameters of radar radiation sources,
identiﬁes their properties, and assesses their threat using
passive reception and signal processing techniques [1], [2].
The ESM receiver’s received radar signal can be viewed
as a stream of roughly random radar pulses made up of
various radar pulse sequences. The source characteristics
andcountermeasurescanonlybedeterminedbysplittingthe
intercepted overlapping pulse streams into pulse sequences
that correspond to various radiation sources. Radar signal
deinterleaving (RSD) is the process of dividing the over-
lapping pulse streams into pulse trains that correspond to
various transmitters, as seen in Fig. 1.
In the pulsed stream of the intercepted radiation source,
the RSD method relies on 5-D characteristics, including
time of arrival (TOA), carrier frequency (RF), pulsewidth
(PW), pulse amplitude (PA), and direction of arrival (DOA).
Schmidt’s work [3] on RSD algorithms began as early as
1974. There are fewer published study ﬁndings since the as-
sociated ﬁeld is sensitive [4], [5], [6], [7], [8], [9], [10], [11].
Currently,RSDmethodscanbecategorizedintothosebased
on traditional pulse description parameters [12] and those
based on machine learning [13]. Based on the remarkable
achievements of deep learning in the ﬁeld of image segmen-
tation and open-set recognition (OSR), this article proposes
an open-set deinterleaving method for radar signals based
on the combination of Laplacian pyramid reconstruction
and reciprocal point adversarial deinterleaving (PR-RPAD).
The PR-RPAD method realizes the feature characterization
from pulse description word (PDW) to pulse description
graph (PDG) through the algorithms of symmetric mapping
of grayscale matrices and Otsu threshold image enlarge-
ment, combined with the Laplacian pyramid multiresolu-
tion feature reconstruction and fusion model. The tower
reshaping from low-resolution to high-resolution features
is accomplished. By introducing the concept of reciprocal
pointstomodeltheopenspaceandproposinganinstantiated
adversarial enhancement method based on the adversarial
network, under the adversarial mechanism between the
reciprocal points and the known radar, the accurate deinter-
leaving of the known and unknown radar radiation sources
in the open space is accomplished. The architecture of the
PR-RPAD method is shown in Fig. 2.
In contrast to conventional RSD techniques [5], [14],
[15], the PR-RPAD algorithm can adapt to complex pulse
repetition interval (PRI) modulation types, performs se-
quence search without requiring traversal of the data, and
11234
IEEE TRANSACTIONS ON AEROSPACE AND ELECTRONIC SYSTEMS
VOL. 61, NO. 5
OCTOBER 2025
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 2]

Fig. 1.
RSD schematic.
Fig. 2.
Architecture of the PR-RPAD algorithm.
does not require multiple rounds of search and merge opera-
tions for radar sources with multiple pulses in a single cycle.
The PR-RPAD approach eliminates the need to iteratively
iterate the input and output data and assign a distinct training
network for every radar radiation source, in contrast to
neural network and automata-based RSD methods [4], [16],
[17], [18]. Using a single network, it is possible to dein-
terleave overlapping pulse streams from several radiation
sources. The PR-RPAD algorithm is capable of accurately
deinterleaving known and unknown radar emitters in the
open space when compared to semantic segmentation dein-
terleaving (SSD) (BLSTM, bidirectional gated recurrent
unit (BGRU), and DCN) [19] techniques.
The contributions of our work can be summarized as
follows.
1) Radar radiation source data processing and con-
version: In this study, the feature characterization
from PDW to PDG for the intercepted radar radia-
tion source PDW data is achieved using the sliding-
window-based gray matrix symmetric mapping ap-
proach. At the same time, the Otsu threshold image
ampliﬁcation technique is used to extend the visual
characterization of pulse description information of
various radar radiation sources and to achieve the
ampliﬁcation and enhancement of PDG edges and
texture structures.
2) Laplacian pyramid feature reconstruction and fu-
sion: This study suggests an architecture for mul-
tiresolution feature reconstruction and fusion that
is based on the Laplacian pyramid. By using jump
connections and multiplication gates, the architec-
ture accomplishes the tower reshaping from low-
resolution to high-resolution features. It also ﬁnishes
the varied resolution characterization of PDG edge
and texture characteristics of radar radiation sources.
3) Reciprocal point adversarial learning mechanism:
This article models the open space using reciprocal
points to lower the risk of closed-set deinterleaving
by confronting known radars. At the same time, an
instantiatedadversarialaugmentationmethodisused
to create confusing training samples, improve the
model’s differentiation against unknown radars, and
ﬁnish the accurate deinterleaving of radar radiation
sources in the open space.
The rest of this article is organized as follows. Section II
introduces RSD, image semantic segmentation (ISS), and
OSR. In Section III, the problem posed in this article is ana-
lyzed and justiﬁed. Based on the analysis in Sections II and
III, the proposed algorithm in Section IV is constructed—an
open-set deinterleaving algorithm for radar signals based on
the combination of Laplacian pyramid reconstruction (PR)
and reciprocal point adversarial learning. Section V designs
relevant experiments and analyzes the experimental results.
Finally, Section VI concludes this article.
II.
RELATED WORK
A.
Radar Signal Deinterleaving
In contemporary electronic warfare, RSD that seeks to
distinguish interleaved pulses from various radar radiation
sources is crucial [20]. Many attempts have been made in
RSD over the past few decades, and these efforts can be
divided into two categories: machine-learning-based ap-
proaches [13] and approaches based on conventional pulse
description parameters [12]. Among other things, pulses
from various radar radiation sources are usually separated
using intrinsic regularities in the TOA sequence in RSD
approaches based on conventional pulse description pa-
rameters. One category of machine-learning-based RSD
LI ET AL.: RADAR SIGNAL OPEN-SET DEINTERLEAVING METHOD
11235
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 3]

TABLE I
Characterization of RSD Algorithms
techniques uses clustering methods, including K-means
[21], alternative fuzzy C-means (AFCM) [22], and density
dynamic clustering [23], to complete the RSD task. The
other category learns discriminative features from tagged
pulses using various deep neural network (DNN) types to
achieve RSD. In light of this, Table I enumerates common
algorithms to RSD and their characteristics.
B.
Semantic Segmentation
Deep convolutional neural networks (DCNNs) have
demonstrated remarkable efﬁcacy in semantic segmentation
and are capable of robustly representing and recognizing
objects and materials by leveraging pretrained feature hi-
erarchies. However since spatial deformation invariance is
introduced in the feature layer, applying DCNN features
to image segmentation results in a decrease in the spatial
resolution of the advanced feature representations, and the
DCNN features do not adequately represent spatial details.
This issue can be resolved by partially recovering the spatial
details lost during the maximum pooling process using un-
poolinganddeconvolution, whichoffersausefulmethodfor
visualizing the feedforward model’s input characteristics.
Higher spatial resolution information is contained in the
feature maps of a DCNN’s lower hierarchical structure, al-
though they lack object category speciﬁcity. Multiple-level
feature maps have been successfully used to classify the
response “jet” in semantic segmentation [38], generic bor-
der detection, synchronization detection and segmentation,
scene recognition, and other applications.
C.
Open-Set Recognition
Scheirer et al. [39] originally presented the OSR prob-
lem and offered a fundamental structure for training and
evaluation following it, drawing inspiration from classiﬁers
with rejection options. Even though OSR is more useful
than popular closed-set classiﬁers, it has been shockingly
overlooked in recent years. OSR research can be catego-
rized into two types: generative models and discriminative
models.
1) Discriminative models: RSD techniques based on
conventional machine learning were put forth before
the application of deep learning. Examples include
support vector machines [40], extreme value ma-
chines [41], open-set nearest neighbor methods [42],
sparse representation OSR methods [43], OpenMax
models [44], and K-sigmoid activation methods [45].
The sigmoid function lacks the compact abating
property in the aforementioned study [40]. If in-
ﬁnitely distant inputs are introduced to all training
data, the aforementioned attribute can be activated;
thus, the open space risk is unlimited.
2) Generative models: In contrast to discriminative
models, generative models create unknown or
known samples using generative adversarial net-
works (GANs) [46], autoencoder (AE) [47], and
ﬂow-based models [48]. This aids the classiﬁer in
determining the decision border between known and
unknown samples. The depth distribution of un-
known classes throughout the learning process is
not taken into account by the aforementioned ap-
proaches, which could result in open space hazards.
III.
DESCRIPTION AND PROCESSING OF PULSES
The aforementioned analysis shows that conventional
deinterleaving algorithms [5], [14], [15] are hard to adapt to
complex PRI modulation styles and need a priori knowledge
guidance. The deinterleaving methods based on neural net-
works [4], [16], [17], [18] are too reliant on a priori informa-
tion and must train the network independently for every kind
ofradiatedsourcesignal.Theaforementionedtechniquesall
rely on time data as their RSD input and are unable to adjust
to complex regime radar signal deinterleaving situations.
Because of this, this work suggests the PR-RPAD algorithm
using the multidimensional data of PDW, including DTOA,
PW, RF, PA, DOA, etc. Through data imageization mapping
and ampliﬁcation enhancement processing, RSD is trans-
formed into ISS. This section describes and analyzes the
PR-RPAD algorithm’s pulse characterization, imageization
mapping, and ampliﬁcation augmentation.
A.
Description of Pulse Characteristics
The determination of pulse description characteristics
is essential for the extraction of each radar’s intercepted
pulses from the entangled pulse streams of several radar
sources and for identifying the radar. Every intercepted
pulse’s RF, PW, PA, TOA, and DOA are measured by ESM.
11236
IEEE TRANSACTIONS ON AEROSPACE AND ELECTRONIC SYSTEMS
VOL. 61, NO. 5
OCTOBER 2025
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 4]

TABLE II
Record Sheet for PDW Parameter
Fig. 3.
Acquisition of PDW data.
As indicated in Table II, every pulse description parameter
is documented in a PDW. Pulse-level radar radiometric
data are essential for creating an efﬁcient radar library for
the ESM and the radar warning system, and PDW data
recording helps evaluate the ESM system’s effectiveness
and grow the electronic intelligence database.
Meanwhile, the radar adjusts the radiation values of
each descriptive parameter in response to the different
tasks performed. This adjustment, at a certain moment, is
manifested in the synergistic cooperation among RF, PW,
and PA; during the detection cycle, it is manifested in the
potential regularity and temporal sequence of the pulse se-
quence, including the asymptotics of DTOA and DOA, the
group variability of RF and PW, and the antenna scanning
characteristics of PA. The descriptive properties of various
radar radiation sources vary from one another. Therefore,
it is possible to extend the differences between the pulse
description parameters of various radar radiation sources
in the pulse sequence and improve the connection between
different parameter types within a single pulse description
parameter by transforming the original pulse description
parameters. This also allows for the visualization of the
positional relationship between the current pulse and the
previous and subsequent pulses.
B.
Pulse Conversion Based on Gray Matrix Mapping
To achieve the visual characterization of the radar PDWs
(TOA, RF, PW, PA, and DOA), as illustrated in Fig. 3,
the DTOA sequence between adjacent pulses is obtained
by performing a ﬁrst-order difference of the intercepted
TOA sequence. The DTOA of the ﬁrst pulse is set to zero,
resulting in the new characterization of the TOA sequence
Fig. 4.
Pulse data chunking using a ﬁxed window length.
and DTOA sequence. A new description of the radar radia-
tion source’s pulse description parameters, namely, DTOA,
RF, PW, PA, and DOA, can be derived by combining the
dimensional pulse description information of RF, PW, PA,
and DOA with the DTOA.
Based on grayscale mapping and the 5-D radar radia-
tion source pulse description parameters (DTOA, RF, PW,
PA, and DOA) mentioned earlier, this article suggests a
grayscale matrix symmetric mapping method based on a
ﬁxed window length to achieve the visual characterization
of radar radiation source pulse description parameters, i.e.,
PDG. This method can extend the differences between
various radar radiation source pulse description parameters
in the pulse sequence and improve the connection between
different parameter types within a single pulse description
parameter. In addition, it visualizes the positional relation-
ship between the current pulse, the preceding pulse, and the
succeeding pulse. The following are the method’s speciﬁc
steps.
1) In the context of radiation source interception, the
radar PDW sequence can be written as follows:
PDW = {PDW1, PDW2, . . . , PDWN} = [DTOA
RF PW PA DOA], where PDWn is the nth PDW,
PDWn = [DTOAn
RFn
PWn
PAn
DOAn], n ∈
{1,
2,
. . . ,
N}; N
is the sequence length.
PDW_Normalized is obtained by normalizing the
DTOA, RF, PW, PA, and DOA in the PDW, re-
spectively. Each pulse description parameter has a
distribution of [0, 1].
2) Set window lengths
lenwindow
and {lenwindow|
lenwindow = 2k+1,
k ∈{0,
1,
2,
. . .}};
then,
use chunks to process PDW_Normalized. When
lenwindow > 1, fake pulses must be added pre-
ceding the start pulse PDW _Normalized1 and
following the end pulse PDW _NormalizedN of
PDW_Normalized. This preserves the consistency
of the data length and format following chunking
and guarantees that the current pulse is always in
the middle of the window length lenwindow. These
fake pulses are deﬁned as duplicates of the start and
end pulses, and the total number of added pulses
is virtualpulse_num = (lenwindow −1)/2. Use Table II
data as an example, and then, process lenwindow = 3
data chunking, as illustrated in Fig. 4.
LI ET AL.: RADAR SIGNAL OPEN-SET DEINTERLEAVING METHOD
11237
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 5]

Fig. 5.
PDW visual description using symmetric mapping of grayscale
matrices with speciﬁed window lengths.
3) Data block tiled follows the chunking process. Each
of the three pulses that make up Pulse Blocks 1–6 in
Fig.4comprisesparameters,suchasDTOA,RF,PW,
PA, and DOA. Thus, to create a vector of length 15, a
single pulse block is tiled. Using Pulse Block 1 as an
example, the following is the expression following
spreading:
PulseBlock1 Expand =
PDW1+
PDW1
PDW2

=
x1
x2
. . .
x15

(1)
where
fake
pulse:
PDW1+ = [DTOA1+
RF1+
PW1+
PA1+
DOA1+ ],
real
pulse:
PDW1 = [DTOA1
RF1
PW1
PA1
DOA1 ], and PDW2 = [DTOA2
RF2
PW2
PA2
DOA2], xi ∈[0, 1], 1 ≤i ≤15.
4) Using a method based on the symmetric mapping
of the grayscale matrix, the PulseBlock1 Expand fol-
lowing the aforementioned Pulse Block 1 pulse
block tiled is projected to the 2-D numerical matrix
PulseBlock1 Matrix using the following expression:
PulseBlock1 Expand
=
⎡
⎢⎢⎢⎣
y1,1
y1,2
· · ·
y1,15
y2,1
y2,2
· · ·
y2,15
...
...
...
...
y15,1
y15,2
· · ·
y15,15
⎤
⎥⎥⎥⎦
(2)
where yi,j = abs(xi−x j )
xmax−xmin , xmax = Max[x1
x2,
. . . ,
x15],
and xmin = Min[x1
x2,
. . . ,
x15].
Using the pulse parameters listed in Table II as an
example, the PDG resulting from symmetric mapping of the
gray matrix for lenwindow = 1 and lenwindow = 3 is displayed
in Fig. 5. The PDGs for the lenwindow = 1 situation are
shown in Fig. 5(a1)–(f1) with resolution 5 × 5. The PDGs
are blurred at this resolution, but we can still determine that
they come from the same radar radiation source because
the feature distributions of the characterization maps in
Fig. 5(c1) and (e1), as well as in Fig. 5(d1) and (f1), are
quite comparable. The PDGs with resolution 15 × 15 for the
lenwindow = 3 condition are shown in Fig. 5(a2)–(f2). PDG
clarity at this resolution is better than that in Fig. 5(a1)–(f1);
however, spurious feature scattering and imprecise charac-
terization of the PDG features in Fig. 5(a2) and (f2) are
caused by the erroneous pulses at both ends of the sequence.
In the PDG of the same radar radiation source, there are
variationsinthefeaturescatterthatareinﬂuencedbythesur-
rounding pulses. As well as taking into account the internal
Fig. 6.
PDG ampliﬁcation and enhancement algorithm ﬂowchart based
on DCCI_Otsu.
storage capacity constraint following PDG production and
the real-time necessity of RSD. Because of this, lenwindow
cannot be inﬁnitely extended merely to improve clarity and
ﬁne feature characterization capabilities and increase the
PDG resolution.
C.
Magniﬁcation and Strengthen of PDG Based on the
Otsu Threshold
Drawing from the aforementioned examination of the
PDW visual characterization outcomes for symmetric map-
ping of ﬁxed window length grayscale matrices, the issues
of low PDG resolution, blurred features, and absent details
in small window lengths, as well as false impulse pertur-
bation, false feature scattering, and imprecise PDG feature
characterization in large window lengths, are addressed. In
addition, the PDG of the same radar source exhibits varia-
tions in feature scattering due to the surrounding impulses.
In this article, we propose an Otsu-threshold-based PDG
ampliﬁcation and enhancement algorithm. This algorithm
is based on directional cubic convolution interpolation, and
the directional cubic convolution interpolation with Otsu
thresholding (DCCI_Otsu) method achieves the magniﬁed
enhancement of PDG edge and texture structure. In other
words, to amplify the PDG regions where there are abrupt
changes in the gray values, and to enhance the PDG regions
where there are regularities, highlighting the variations,
potential combinatorial relationships, and pulse-to-pulse
temporal connections among radar radiation source pulse
description parameters extends the visual representation
of pulse description information of various radar radiation
sources and produces more stable and better-magniﬁed
images in terms of edge and texture feature retention.
Fig. 6 depicts the operation ﬂow in detail. Together with
the pertinent descriptions found in the literature [49], the
Otsu threshold t∗is ﬁrst computed for the low-resolution
PDG, LR, followed by the extension of the LR to the
high Resolution (HR) PDG, Y , the calculation of the three
convolutions in the 45° and 135° directions, respectively,
11238
IEEE TRANSACTIONS ON AEROSPACE AND ELECTRONIC SYSTEMS
VOL. 61, NO. 5
OCTOBER 2025
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 6]

Fig. 7.
PDG ampliﬁcation and enhancement at 65 × 65 resolution.
and the determination of the missing pixel p, to achieve the
PDG ampliﬁcation and enhancement.
In conclusion, using Fig. 5(a1)–(f1) as an example,
the original PDGs are ampliﬁed and enhanced using the
DCCI_Otsu-based PDG ampliﬁcation and enhancement
method; the outcomes are displayed in Fig. 7. The PDG
in Fig. 7 is of 65 × 65 resolution, and when compared
to the 5 × 5 resolution PDG in Fig. 5, it has better char-
acterization ﬁndings for the same category of radar ra-
diation source description characteristics, more delicate
detail characterization, and much-improved clarity. How-
ever, the DCCI_Otsu algorithm’s time complexity rises
with the PDGs resolution, and the memory needed to
store the PDGs increases geometrically. These factors
make it unfavorable for the RSD algorithm’s eventual
implementation.
IV.
MODEL CONSTRUCTION
The DCCI_Otsu image ampliﬁcation and enhancement
technique and ﬁxed window length gray matrix symmetric
mapping are used to achieve the goal of a high-quality
radar radiation source impulse description parameter visu-
alization representation image. In this article, we propose
a multiresolution feature reconstruction and fusion model
based on the Laplacian pyramid, which mitigates the “spa-
tial semantic uncertainty” phenomenon in the traditional
convolutional feature maps, uses jump connections and
multiplication gates to realize the tower reshaping from
low-resolution to high-resolution features, and completes
the multiresolution characterization of PDG edge and tex-
ture features. The open space is modeled by introducing
the idea of reciprocal points from the standpoint of multi-
class integration. Confrontation with known radars lowers
the risk of closed-set deinterleaving; to ﬁnish the accurate
deinterleaving of radar radiation sources in the open space
and improve the discriminatory nature of the model against
unknown radars, confusion samples are generated for train-
ing under the confrontation mechanism between reciprocal
points and known radars using an instantiated confrontation
enhancement method.
Fig. 8.
Architecture of Laplacian pyramid fusion and reconstruction.
A.
Laplacian PR Model
A Laplacian-pyramid-based multiresolution feature re-
construction and fusion model is proposed, which draws
inspiration from residual networks. Fig. 8 shows the general
structure of the Laplacian pyramid multiresolution feature
reconstruction and fusion model. The spatial accuracy of the
segmentation masks is increased by performing a series of
successive reconstructions starting with coarse-scale “low-
frequency (it refers to PDG edge information, meaning
the region where the PDG gray value quickly changes)”
segmentation estimations and adding data from “high-
frequency (it refers to PDG’s textural structure information,
meaning the region that exhibits regularity)” subbands.
The main body uses the VGG architecture feature ex-
tractionnetworktoextract3-Dfeatures, 16 × 16 × 128, 8 ×
8 × 512, 6 × 6 × 4096, after ﬁrst applying the PDG with
resolution 65 × 65, as illustrated in Fig. 8. The 16 × 16 ×
128, 8 × 8 × 512, and 6 × 6 × 4096 feature blocks are then
reconstructed into A1 (65 × 65 × 4), A2 (32 × 32 × 4),
and A3 (16 × 16 × 4) score maps using the Reconstruction
module, which is built using an inverse convolutional net-
work. The number four indicates the number of radar radia-
tion sources. Then, A3 and F2 (32 × 32 × 4) are upsampled,
respectively, to achieve the fusion between various resolu-
tion pulse description score maps, yielding the upsampled
outputs B2 (32 × 32 × 4) and B1 (65 × 65 × 4). After that,
B2 and B1 are fed into Softmax to produce C2 and C1, which
are subsequently put into the Mask module, which produces
D2 and D1 by deducting the “low-frequency” components
from the high-resolution feature map. D2 and D1 are then
fed into the Elementwise Product module, where they are
multiplied element by element with the original feature
maps A2 and A1 to produce E2 and E1, or the results of
the primary fusion of the features. Finally, B2 and B1 are
added together with E2 and E1 to produce F2 and F1, or
the outcomes of the primary fusion of the features. The
secondary feature fusion results, or F2 and F1, are then
obtained by adding B2 and B1 to E2 and E1. The feature
reconstruction and fusion at the 1.0, 0.5, and 0.25 PDG
scales are ﬁnally ﬁnished using F1, F2, and A3 as the outputs
LI ET AL.: RADAR SIGNAL OPEN-SET DEINTERLEAVING METHOD
11239
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 7]

of the Laplacian pyramid multiresolution reconstruction
and fusion module.
B.
Open-Set Networks for Reciprocal Point Adversarial
Learning
In
a
set
of
n
labeled
samples
DL =
{(x1, y1), . . . ,(xn, yn)},
there
are
N
known
radiation
sources
(yi ∈{1, . . . , N}
is
the
label
of
xi)
and
a
greater
amount
of
test
data
DT = {t1, . . . ,tu}
({1, . . . , N} ∪{N + 1, . . . , N + U} is labeled ti), with
U being the number of unknown radiation sources in the
real scenario. The open space Opos
k
from other known
radiation sources and the ﬁnite unknown space Oneg
k
are the
two subspaces of Ok, assuming that the deep embedding
space of category k is Sk and its corresponding open space
is Ok, i.e., Ok = Opos
k
∪Oneg
k .
1) Reciprocal Point Adversarial Learning: This article
deﬁnes Du as a sample from outside DL, and Dk
L ∈Sk,
D̸=k
L
∈Opos
k , and Du ∈Oneg
k
as positive training data, nega-
tive training data, and perhaps unknown data, respectively.
Furthermore, the reciprocal point Pk of the category k
is thought to be a possible representation of D̸=k
L ∪Du.
Accordingly, Ok is nearer the reciprocal point Pk than Sk,
using the following formula:
max

ζ

D̸=k
k
∪Du, Pk
≤d
∀d ∈ζ

Dk
L, Pk
(3)
where ζ (·, ·) represents all the sample distance between the
computational sets.
The reciprocal points and the associated known cate-
gories have a dyadic relationship that can be used to execute
RSD based on the aforementioned equation. To be more
precise, a deep embedding function C with a learnable
parameter θ can optimize the reciprocal points of a category
and represent them with m dimensions. Combining the
Euclidean distance de with the dot product dd yields the
distance d(C(x), Pk), given a sample x and a reciprocal
point Pk
de

C (x) , Pk
= 1
m 
C (x) −Pk2
2
(4)
dd

C (x) , Pk
= C (x)  Pk
(5)
d

C (x) , Pk
= de

C (x) , Pk
−dd

C (x) , Pk
.
(6)
The aforementioned formula is used to estimate the
differencebetweentheembeddedfeatureC(x)andtherecip-
rocal points of the known category to identify the category
to which it belongs. The process is then normalized using
the Softmax function based on the property that the total is
1, yielding the ﬁnal classiﬁcation probability
p

y = k |x, C, P

=
eγ d(C(x),Pk)
N
i=1 eγ d(C(x),Pi)
(7)
where the ease of distance-probability conversion is con-
trolled by the hyperparameter γ .
To optimize the learnable parameter θ in the deep em-
bedding function C, the reciprocal point classiﬁcation loss
Fig. 9.
Network framework for the reciprocal point adversarial learning
open set.
based on the true category k negative logarithmic proba-
bility is minimized. In addition, the empirical classiﬁcation
risk is decreased by minimizing the following reciprocal
point classiﬁcation loss:
Lc (x; θ, P) = −log p

y = k |x, C, P

.
(8)
Apart from categorizing the known categories, the fol-
lowing equation illustrates how the known and unknown
spaces are divided by optimizing the distance between the
reciprocal points and the matching training samples:
arg max
f ∈H

ζ

Dk
L, Pk
.
(9)
The formulation that follows demonstrates how the
complementary nature of Sk and Ok is used to indirectly
constrain the open space by limiting the distance between
Sk and the reciprocal point Pk to be less than R:
Lo

x; θ, Pk, Rk
= max

de

C (x) , Pk
−R, 0

(10)
where R is a learnable parameter; a wider range of non-k
samples is obtained using the Euclidean distance.
The total loss function in reciprocal point adversarial
learning integrates (8) and (10) and can manage both the
open space risk and the empirical classiﬁcation risk
L

x, y; θ, P, R

= Lc (x; θ, P) + λLo (x; θ, P, R) (11)
where θ, P, and R are the learnable parameters and λ is the
open space risk module weight.
2) Instantiate Adversarial Enhancement: Although
such classiﬁers can differentiate unknown distributions
without any prior knowledge of the unknown data, they are
vulnerable to perplexing samples produced by basic gener-
ators. To enhance the classiﬁer’s discriminative capacity for
unknown distributions, a generative network is employed to
provide more data as confusing samples (CSs) for unknown
data DU.
In this research, rather than recovering known samples
Sk, we employ a generator to recover partial CS from
the global open space OG. An adversarial enhancement
framework comprising a discriminator D, a generator G,
and a classiﬁer C with a deep embedding function C is
created and shown in Fig. 9. The classiﬁer C is used to
discriminate the likelihood that a sample belongs to a known
class. The latent variable z in the prior distribution Ppri(z)
is mapped to the output G(z) by the generator G. After
that, the discriminator D is tuned to distinguish between
11240
IEEE TRANSACTIONS ON AEROSPACE AND ELECTRONIC SYSTEMS
VOL. 61, NO. 5
OCTOBER 2025
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 8]

true and generated samples using {z1, . . . , zn} in Ppri(z)
and known samples {x1, . . . , xn}.
In this study, we propose an adversarial process between
reciprocal points and known classes that motivates the
generator to generate samples close to each open space
Ok center Pk, i.e., the generated PDG is motivated to be
close to the global open space OG. Instead, to confuse the
discriminatorD,thegeneratorGanticipatessamplesthatare
more similar to the recognized categories. When the two
adversarial methods mentioned previously are combined,
the generator G is optimized using the following formula:
max
G
1
n
n

i=1

log D (G (zi)) + β  H (zi, P)
(12)
where H(zi, P) = −1
N
N
k=1 S(zi, Pk)  log(S(zi, Pk)) is
the information entropy function and β is the hyperparam-
eter that regulates the information entropy’s loss weight.
To train the feature space, utilize the generated samples
as unknown data DU. Then, optimize the classiﬁer C using
the following formula to minimize the open space OG size,
which lowers the open space risk:
min
C
1
n
n

i=1

L

xi, yi

−β · H (zi, P)
(13)
where L represents adversarial learning overall loss at re-
ciprocal points.
The aforementioned process, the auxiliary batch nor-
malization method, stops the CS from having a detrimental
effect on the discriminations between known categories.
Finally, mutually reinforcing joint training between the
discriminator D and the classiﬁer C and the CS generator
G using an alternating algorithm optimizes the aforemen-
tioned goals. By training the classiﬁer C with the known
categories after the CS trains the classiﬁer C, the classiﬁer
C is encouraged to concentrate on the known categories and
compensates for the CS bias.
C.
Pyramid Reconstruction and Reciprocal Point Adver-
sarial Deinterleaving
The general ﬂow of the reciprocal point adversarial
learningandLaplacianPR-basedradarsignalopen-setdein-
terleaving approach presented in this article is illustrated
in Fig. 10. Using a three-level Laplacian PR and fusion
model, the PR-RPAD model uses the PDG with resolution
size 65 × 65 as input and produces feature score maps with
resolutions 65 × 65 × N, 32 × 32 × N, and 16 × 16 × N
from top to bottom. The number of known radar sources
is N. Consequently, reciprocal point adversarial learning
generates three learning processes, each of which is optimal
under the loss function displayed in (11). The overall loss
function of the PR-RPAD method is obtained by taking a
weighted sum of the learning losses of the various processes
LPR−RPAD = α1L65×65×N + α2L32×32×N + α3L16×16×N
(14)
where L65×65×N, L32×32×N, and L16×16×N are the reciprocal
point adversarial learning loss functions displayed in vari-
ous processes for (11); α1, α2, and α3 are the loss function
weight hyperparameters.
V.
EXPERIMENTAL DETAILS
Simulation tests are carried out in this section to verify
the PR-RPAD algorithm deinterleaving capability in the
presence of known and unknown mixed radar radiation
source signals. The conﬁguration of the measurement in-
terception scenario for the simulation experiments is de-
scribed in Section V-A. Details of the simulation data pa-
rameter settings and experimental distribution are provided
in Section V-B. Evaluation metrics, control experiments,
training methodology, and dataset setup are all covered in
Section V-C. In Section V-D, the deinterleaving outcomes
of the PR-RPAD algorithm are described and examined.
A.
Data Simulation
This work establishes two radar radiation source mea-
surement interception scenarios based on the PR-RPAD
algorithm model characteristics and the description of the
radar radiation source pulse parameters in Section III-A.
Scenarios 1 and 2 include the interception of radar radiation
source measurements in closed and open spaces.
In Scenario 1, the intercept pulse data simulation using
(a), (b), and (c) is measured using three radar radiation
sources. The simulation’s difﬁculty and data complexity are
growing steadily. The details are listed as follows.
a) Measurements intercept pulse data simulation for
single-function radar radiation sources. All of the
radar radiation sources in these simulation data are
single-function radars, each of which has a single set
of radiation parameters. The data only show three
PRI modulation styles: Fixed, Staggered, and D&S.
b) Measurements intercept pulse data simulation for
multifunction radar radiation sources. Both single-
function and multifunction radars are depicted in
these simulated data; the multifunction radars are
freely deployable based on the mission mode and
have two or more sets of radiation parameters. The
data only show three PRI modulation styles: Fixed,
Staggered, and D&S. Simulated data simultaneously
show radiated sources of the same type of radar with
the same RF, PW, and PRI, with the only differences
being in DOA and PA.
c) Measurements intercept pulse data simulation for
radar radiation sources under jittered PRI. Both
single-functionandmultifunctionradarsaredepicted
in these simulated data; the multifunction radars
are freely deployable based on the mission mode
and have two or more sets of radiation parameters.
The data exhibit four PRI modulation styles: Fixed,
Staggered, Jittered, and D&S. Simulated data simul-
taneously show radiated sources of the same type of
radar with the same RF, PW, and PRI, with the only
differences being in DOA and PA.
LI ET AL.: RADAR SIGNAL OPEN-SET DEINTERLEAVING METHOD
11241
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 9]

Fig. 10.
PR-RPAD method.
In scenario 2, the complexity and difﬁculty of the data
gradually rise when three different types of radar radia-
tion sources—(d), (e), and (f)—are put up to mimic the
intercepted pulse data. Corresponding to (a), (b), and (c)
in Scenario 1, respectively, and to test the PR-RPAD algo-
rithm’s open-set deinterleaving capability, many radar radi-
ation sources were randomly assigned as unknown radiation
sources.
It is necessary to take into account pulse loss and random
noise brought on by modiﬁcations to the measurement
interception environment in the data simulation above. The
pulse loss rate is denoted as follows:
ρl = Nloss
Nsum
(15)
where Nsum represents the total number of pulses from the
radiation source and Nloss represents the number of pulses
lost from the source.
ρn is the ratio of the average number of target pulses to
the number of random noise, i.e.,
ρn = Nnoise
Naverage
(16)
where Naverage is the average number of pulses from the
radiation source and Nnoise is the number of random noises.
ρn/(ρn + C) can be used to determine the ratio of random
noise to total pulses, while C represents the number of
radiation sources.
11242
IEEE TRANSACTIONS ON AEROSPACE AND ELECTRONIC SYSTEMS
VOL. 61, NO. 5
OCTOBER 2025
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 10]

TABLE III
Radar Radiation Source Parameters for Experiment 1
TABLE IV
Radar Radiation Source Parameters for Experiment 2
B.
Design of Experiments
The experiments that follow are intended to conﬁrm
the PR-RPAD algorithm viability and the deinterleaving’s
performance using the six simulated data situations in the
two scenarios mentioned earlier. Among them, Experiments
1–3 match the data simulation’s (a)–(c); Experiments 4–6
match the data simulation’s (d)–(f). All experimental data
ρl and ρn are chosen at random from a predetermined
range. The amplitude data simulation is produced in the
data-generating process of the aforementioned experiment
by adjusting the radar radiation source antenna’s scanning
pattern and the range of its amplitude value ups and downs.
In addition, mechanical scanning is M, 1-D phase scanning
is P1, and 2-D phase scanning is P2. The full design of the
experimental data is as follows (Known: K, Unknown: U;
Fixed: F, Agile: A, Stagger: S, D&S: DS, and Jitter: J).
Experiment 1: A single-function radar radiation source
in closed-set space measures the deinterleaving of inter-
cepted pulse signals. The parameters for the radar radia-
tion source are conﬁgured as indicated in Table III, where
0 ≤ρl ≤0.5 and 0 ≤ρn ≤0.5.
Experiment 2: A multifunction radar radiation source in
closed-set space measures the deinterleaving of intercepted
pulse signals. The parameters for the radar radiation source
are conﬁgured, as indicated in Table IV, where radiation
sources 1 and 3 are radiation sources of the same type and
TABLE V
Radar Radiation Source Parameters for Experiment 3
TABLE VI
Radar Radiation Source Parameters for Experiment 4
radiation source 5 is a multifunction radiation source. 0 ≤
ρl ≤0.5 and 0 ≤ρn ≤0.5.
Experiment 3: A radar radiation source with jitted PRI in
closed-set space measures the deinterleaving of intercepted
pulse signals. The parameters for the radar radiation source
are conﬁgured, as indicated in Table V, where radiation
sources 1 and 3 are radiation sources of the same type
and radiation sources 2 and 5 are multifunction radiation
sources. 0 ≤ρl ≤0.5 and 0 ≤ρn ≤0.5.
Experiment 4: A single-function radar radiation source
in an open-set space measures the deinterleaving of inter-
cepted pulse signals. The parameters for the radar radiation
source are conﬁgured, as indicated in Table VI, where
radiation source 2 is speciﬁed as an unknown radiation
source; 0 ≤ρl ≤0.5, and 0 ≤ρn ≤0.5.
Experiment 5: A multifunction radar radiation source
in an open-set space measures the deinterleaving of inter-
cepted pulse signals. The parameters for the radar radiation
LI ET AL.: RADAR SIGNAL OPEN-SET DEINTERLEAVING METHOD
11243
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 11]

TABLE VII
Radar Radiation Source Parameters for Experiment 5
TABLE VIII
Radar Radiation Source Parameters for Experiment 6
source are conﬁgured, as indicated in Table VII, where
radiation sources 1 and 3 are radiation sources of the same
type, radiation source 5 is a multifunction radiation source,
and radiation sources 2 and 3 are speciﬁed as unknown
radiation sources; 0 ≤ρl ≤0.5 and 0 ≤ρn ≤0.5.
Experiment 6: A radar radiation source with jitted PRI
in an open-set space measures the deinterleaving of inter-
cepted pulse signals. The parameters for the radar radiation
source are conﬁgured, as indicated in Table VIII, where
radiation sources 1 and 3 are radiation sources of the same
type, radiation sources 2 and 5 are multifunction radiation
sources, and radiation sources 2, 3, and 6 are speciﬁed as un-
known radiation sources; 0 ≤ρl ≤0.5 and 0 ≤ρn ≤0.5.
C.
Datasets, Training Methods, Control Experiments,
and Performance Metrics
1) Dataset: In this article, two types of datasets—a
mixed dataset and a particular intercepted environment
dataset—are produced during the experiments. In each ex-
periment, there is only one hybrid dataset, which is made
up of PDW sequences with a sample size of 200 and a
length of 400 per sample for training and validating RSD
models. ρl and ρn are split among [0, 0.5]. There are two
categories into which speciﬁc-intercept scenario datasets
canbedivided: 1⃝thereareﬁvedatasetswithmeasurements
and intercepts of reconnaissance receivers in good condi-
tion (with a pulse loss rate of ρl = {0}) and interception
environments that are gradually getting worse (with noise
ratios of ρn = {0.1, 0.2, 0.3, 0.4, 0.5}) and 2⃝in intercep-
tion status and reconnaissance receiver measurements, the
interception environment progressively deteriorates, and
the impulse loss rate equals the noise ratio, i.e., ﬁve datasets
ρl = ρn = {0.1, 0.2, 0.3, 0.4, 0.5}. The aforementioned ten
datasets, each including 100 PDW sequences with a length
of 400, are used to test the RSD algorithm’s performance
under various measurement interception conditions.
2) PR-RPAD Model Training Approach:
1⃝Closed-set
scenario training: The training dataset, validation dataset,
and test dataset are separated in an 8:1:1 ratio. The PR-
RPAD model was trained in an environment with Windows
10 22H2, Python 3.8.16 64-bit, and Torch 2.0.1+cu117.
With 100 iterations and a learning rate of 0.01, the stochas-
tic gradient descent (SGD) optimizer was employed. No
instantiation enhancement mechanism is added, and GANs
are not required to synthesize obfuscated samples.
2⃝Open-set scenario training: Within the hybrid
dataset, the known radiation source training, validation, and
test datasets (Test Dataset) are split in an 8:1:1 ratio. The
unknown radiation source data are set as the unknown class
dataset (Out Dataset). The Test Dataset and the Out Dataset
are utilized for evaluating the model after training. The PR-
RPAD model was trained in an environment with Windows
10 22H2, Python 3.8.16 64 bit, and Torch 2.0.1+cu117.
With 100 iterations and a learning rate of 0.01, the SGD
optimizer was employed. The confused sample generation
network learning rate is 0.0002, and the Adam optimizer
was employed.
In conclusion, the real-time RSD requirement and the
internal storage capacity limitation following PDG gener-
ation are taken into account during the PR-RPAD model
training and testing process. The PDGs were generated in
all tests under the lenwindow = 1 condition that the model
inputs were PDG-enhanced and magniﬁed images with a
65 × 65 resolution. To achieve RSD with a reduced time
cost and less memory space, the captured interleaved pulse
data are fed as a stream and processed sequentially, ﬁnally
freeing the cache and outputting the result.
3) Control Experiments:
1⃝Closed-set scenario:
BLSTM, BGRU, DCN, SDIF, and PRI-Tran.
2⃝Open-set scenario: SDIF and PRI-Tran.
The forward and reverse DTOA information is equal in
RSD. To achieve RSD, the literature [19] uses bidirectional
11244
IEEE TRANSACTIONS ON AEROSPACE AND ELECTRONIC SYSTEMS
VOL. 61, NO. 5
OCTOBER 2025
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 12]

TABLE IX
Experiment 1: Analysis of Model Performance on a Mixed Dataset
(Closed Set)
TABLE X
Experiment 2: Analysis of Model Performance on a Mixed Dataset
(Closed Set)
recurrent neural network (BRNN) for DTOA sequence pro-
cessing and connects the BRNN step output with a fully
connected layer. The study employs the BGRU and BLSTM
recurrent neural network (RNN) architectures, and it makes
use of bidirectional recurrent neural network (TCN)s’ pop-
ularity in sequence modeling [50] to achieve RSD. In
this study, we choose the SDIF and PRI-Tran methods as
the control experiments and establish the threshold value
to ﬁnish the sequence search based on the literature. To
determine the number of detected pulses throughout the
search, the initial pulse is utilized as the beginning point; if
it surpasses the threshold, it is the target pulse. Otherwise,
the search fails. Then, the following pulse is used as the
search’s beginning point, and so on. As a result, the SDIF
and PRI-Tran techniques’ deinterleaving process does not
need differentiating between known and unknown radars.
The algorithm’s closed-set and open-set indexes, however,
ought to be determined using the known and unknown
radar sets listed in the associated experimental parameter
table.
4) Performance Metrics:
1⃝Closed-set scenario: ac-
curacy (ACCclosed−set), mean recall (MR), mean precision
(MP), and mean intersection over union (MIOU).
2⃝Open-set scenario: accuracy (ACCknown), and open-
set classiﬁcation rate (OSCR) [51].
D.
Results
1) The model performance analysis for Experiments 1–
6 under the mixed dataset is displayed in Tables IX–
XI. The loss versus accuracy curves in the model’s
training process from Experiments 1–3 are plotted
as illustrated in Fig. 11 to examine the PR-RPAD
TABLE XI
Experiment 3: Analysis of Model Performance on a Mixed Dataset
(Closed Set)
Fig. 11.
(a) and (b) Training loss versus accuracy curves for the
PR-RPAD model under mixed datasets, closed-set scenario
Experiments 1–3.
algorithm’s overall performance and convergence in
the closed-set situation. Among these, Fig. 11(a)
displays the loss curve during the PR-RPAD model’s
training process; it is evident from the ﬁgure that the
loss curve steadily declines as the number of train-
ing iterations rises and tends to stabilize at roughly
40 iterations. In Experiments 1–3, both stabilize at
roughly 40 iterations, although the loss rises as the
dataset’s complexity increases. The accuracy curve
during the PR-RPAD model’s training process is
depicted in Fig. 11(b); it is evident from the ﬁgure
that the accuracy progressively rises as the number
of training iterations grows, and the accuracy curve
tends to stabilize at roughly 60 iterations. In Exper-
iments 1–3, the accuracy rate has dropped as the
dataset’scomplexityhasincreased,butallrepetitions
have stabilized at roughly 60 times. In conclusion,
Experiments 1–3 of the closed-set scenario show
LI ET AL.: RADAR SIGNAL OPEN-SET DEINTERLEAVING METHOD
11245
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 13]

Fig. 12.
(a)–(c) RSD confusion matrix for the mixed dataset of closed-set scenarios in Experiments 1–3.
that the accuracy rate declines and the loss of the
PR-RPAD model grows as the dataset complexity
increases; however, the curve tends to stabilize and
the model converges at roughly 40–60 iterations on
average.
As can be seen from Tables IX–XI, the model perfor-
mance analysis results of Experiments 1–3 in the closed-
set scenario, the PR-RPAD algorithm completely outper-
forms the traditional algorithms, such as SDIF, PRI-Tran,
BLSTM, BGRU, and DCN, in the metrics of ACCclosed−set,
MR, MP, and MIOU. One of the main causes of the
comparison algorithm’s poor performance is the lack of
enough training data, and the quantity of training samples
used in this work is not in line with previous research
[19]. In addition, the PR-RPAD algorithm receives ﬂawless
scores on every index because there are not many radar
radiation sources in Experiment 1. The PR-RPAD model
performs well in terms of RSD and noise rejection, as seen
in Fig. 12(a). As the dataset became more complicated
in Experiment 2, the PR-RPAD model’s deinterleaving
performance declined. The PR-RPAD model continues to
exhibit good performance in terms of partial radar radiation
source deinterleaving and noise rejection, as illustrated in
Fig. 12(b). For radar radiation sources 1 and 3 of the same
model, with the same parameters of RF, PW, and PRI, the
RSD features can only be extracted from the DOA and PA
information. Due to the interleaving of the DOA and PA
information of radar radiation sources 1 and 3, the RSD
performance is slightly degraded. In Experiment 3, the
PR-RPAD model’s deinterleaving performance deteriorates
much more as a result of the dataset’s increased complexity.
The PR-RPAD model continues to perform well in partial
radar radiation source deinterleaving and noise rejection, as
illustratedinFig.12(c).Theradartypeofradiationsources1
and 3 remains the same, and the more radiation sources there
are in the scenario, the more challenging RSD becomes. As
aresult,inExperiment3,thedeinterleavingperformancefor
the identical model of radiation sources drastically declines.
Fig. 13 plots the loss versus accuracy curves for the
model’s training process for Experiments 4–6 in the open-
set scenario. Among these, the loss curve during the PR-
RPAD model’s training process is depicted in Fig. 13(a);
Fig. 13.
(a) and (b) Training loss versus accuracy curves for the
PR-RPAD model under mixed datasets, open-set scenario Experiments
4–6.
it is evident from the ﬁgure that the loss curve steadily
declines as the number of training iterations increases and
stabilizes at roughly 40 iterations. In Experiments 4–6,
both stabilize at roughly 60 iterations, although the loss
rises as the dataset’s complexity increases. According to
Fig. 13(b), which depicts the accuracy curve during the
PR-RPAD model’s training process, the accuracy progres-
sively rises as the number of training iterations grows,
and the accuracy curve tends to stabilize at roughly 40
iterations. The accuracy rate dropped in Experiments 4–6
as the dataset’s complexity increased, but both iterations
stabilized at roughly 60 times. The PR-RPAD model loss
rose and the accuracy rate declined with increasing dataset
complexity in Experiments 4–6 of the open-set scenario.
However, after 40–60 iterations, the curve tends to settle
and the model converges.
11246
IEEE TRANSACTIONS ON AEROSPACE AND ELECTRONIC SYSTEMS
VOL. 61, NO. 5
OCTOBER 2025
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 14]

Fig. 14.
(a)–(c) RSD confusion matrix for the mixed dataset of open-set scenarios in Experiments 4–6.
TABLE XII
Experiments 4–6: Model Performance Analysis With Mixed Datasets
(Open Set)
In terms of ACCknown, OSCR, and other metrics, the
PR-RPAD method substantially outperforms conventional
algorithms like SDIF and PRI-Tran, according to the ﬁnd-
ings of model performance analysis for Experiments 4–6 in
the open-set scenario, as shown in Table XII. Among these,
the SDIF and PRI-Tran algorithms hunt for the regularity
of the TOA sequence to complete RSD, which makes dein-
terleaving challenging in intricate PRI modulation settings.
Meanwhile, the PR-RPAD algorithm has achieved ﬂawless
scores in all indexes because Experiment 4 had a limited
number of radar radiation sources. The PR-RPAD model
performs better in terms of noise rejection and RSD, as seen
in Fig. 14(a). Experiment 5 shows that the PR-RPAD model
performs better in known radar radiation source deinterleav-
ing due to the dataset’s increasing complexity, whereas its
performance declines in unknown radar radiation source
deinterleaving. As illustrated in Fig. 14(b), the PR-RPAD
model continues to exhibit good RSD and noise rejection
performance. As the complexity of the dataset increases
in Experiment 6, the PR-RPAD model’s deinterleaving
performance deteriorates even more. The PR-RPAD model
continues to exhibit good performance in terms of partial
radar radiation source deinterleaving and noise rejection,
as illustrated in Fig. 14(c). Radiation sources 1 and 5 in-
terleave in terms of radiation parameters, which degrades
the performance of the PR-RPAD model. In conclusion,
the PR-RPAD model’s performance steadily declines as
the number of unknown radiation sources and parameter
Fig. 15.
RSD curves under speciﬁc interception conditions in
Experiment 1. (a) Pulse loss rate equals zero. (b) Pulse loss rate equals
the noise ratio.
complexity rises for the open-set scenario Experiments 4–6,
and the OSCR index also declines.
2) Figs. 15–20 demonstrate the model performance
analysis for Experiments 1–6 under particular inter-
ception scenario datasets. In the closed-set scenario,
as illustrated in Figs. 15–17, the PR-RPAD algo-
rithm outperforms both traditional techniques like
SDIF and PRI-Tran and deep learning techniques
like BLSTM, BGRU, and DCN in the case of a good
reconnaissance measurement interception environ-
ment and a gradual deterioration of the reconnais-
sance measurement interception environment. It is
also highly robust and unaffected by changes in the
reconnaissance measurement interception environ-
ment.
LI ET AL.: RADAR SIGNAL OPEN-SET DEINTERLEAVING METHOD
11247
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 15]

Fig. 16.
RSD curves under speciﬁc interception conditions in
Experiment 2. (a) Pulse loss rate equals zero. (b) Pulse loss rate equals
the noise ratio.
Fig. 17.
RSD curves under speciﬁc interception conditions in
Experiment 3. (a) Pulse loss rate equals zero. (b) Pulse loss rate equals
the noise ratio.
Fig. 18.
RSD curves under speciﬁc interception conditions in
Experiment 4. (a) Pulse loss rate equals zero. (b) Pulse loss rate equals
the noise ratio.
Fig. 19.
RSD curves under speciﬁc interception conditions in
Experiment 5. (a) Pulse loss rate equals zero. (b) Pulse loss rate equals
the noise ratio.
11248
IEEE TRANSACTIONS ON AEROSPACE AND ELECTRONIC SYSTEMS
VOL. 61, NO. 5
OCTOBER 2025
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 16]

Fig. 20.
RSD curves under speciﬁc interception conditions in
Experiment 6. (a) Pulse loss rate equals zero. (b) Pulse loss rate equals
the noise ratio.
Meanwhile, we believe that the study of the deinterleav-
ing algorithm’s performance should compute its capacity to
differentiate between noise, clutter, and interference signals
inadditiontoassessingitsdeinterleavingperformancetothe
target radiation source pulse. As a result, in conventional
RSD algorithms like SDIF and PRI-Tran, the noise in the
radiation pulse stream gradually increases as the reconnais-
sance measurement interception environment changes. This
results in a diluted density of the target radiation source
pulse, which is then correctly extracted with the help of
the algorithm’s strong noise rejection capability, and an
overallupwardtrendinRSDperformance.Theperformance
curve displays an up-and-down phenomenon, and the noise
in the radiation pulse stream gradually increases with the
change of the reconnaissance measurement interception
environment in deep learning RSD algorithms like BLSTM,
BGRU, DCN, etc. This leads to unstable RSD performance,
poor robustness, and an increased risk of deinterleaving.
In the open-set scenario, as illustrated in Figs. 18–20,
the PR-RPAD algorithm outperforms traditional techniques
like SDIF and PRI-Tran in known radar radiation source
deinterleaving under both good and gradually deteriorating
reconnaissance measurement interception environments.
It also exhibits strong robustness and is not affected by
changes in the reconnaissance measurement interception
environment. The PR-RPAD model’s performance in Ex-
periment 4 is stable with minimal ﬂuctuation in the data
from an unknown radar radiation source. In Experiments
5 and 6, the noise in the radiation pulse stream gradually
increases as the reconnaissance measurement interception
Fig. 21.
(a) and (b) Experiment 3: PR-RPAD model training loss and
accuracy curves with varying weight values.
environment changes, but the total amount of data stays the
same. Furthermore, the description parameters for the target
radiation source pulse and noise differ signiﬁcantly. The
PR-RPAD algorithm’s strong noise discrimination capabil-
ity supports the target radiation source pulses’ proper dein-
terleaving and the overall RSD accuracy’s upward trend.
In SDIF and PRI-Tran, as the reconnaissance measurement
intercept environment changes, i.e., the proportion of noise
increases, leading to DTOA anomalies; at the same time,
the loss rate increases, leading to a decrease in radiation
pulse density. The algorithms SDIF and PRI-Tran mine
the potential laws of TOA sequences to gradually remove
noise and dilute the intercepted sequences. Finally, the
target radar sequence is extracted from the low-density
intercept sequence to achieve the overall improvement of
RSD accuracy.
3) The impact of the loss function weights on
the PR-RPAD model’s performance is examined
using the loss function weights α1, α2, and
α3 of (1, 0, 0), (0, 1, 0), (0, 0, 1), (1/2, 1/2, 0),
(1/2, 0, 1/2), (0, 1/2, 1/2), and (1/3, 1/3, 1/3), re-
spectively, to examine the RSD performance of the
PR-RPAD model in Experiments 3 and 6. Fig. 21
displays the PR-RPAD model’s training loss ver-
sus accuracy curves for Experiment 3 using the
weights mentioned earlier. The loss curve for weight
α1 = α2 = α3 = 1/3 converges more quickly than
LI ET AL.: RADAR SIGNAL OPEN-SET DEINTERLEAVING METHOD
11249
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 17]

Fig. 22.
(a) and (b) Experiment 6: PR-RPAD model training loss and
accuracy curves with varying weight values.
the other cases in Fig. 21(a). The accuracy curve
for weight α1 = α2 = α3 = 1/3 is higher than that
of other cases in Fig. 21(b). Thus, the PR-RPAD
model’s RSD performance may be maximized by ad-
justing the loss weight α1 = α2 = α3 = 1/3 on vari-
ous pyramid-level feature reconstruction outcomes.
Fig. 22 displays the PR-RPAD model’s training loss
versus accuracy curves for Experiment 6 using the weights
mentioned above. The pace of convergence of the PR-
RPAD model loss curves is not signiﬁcantly impacted by
the various weight choices in the open-set scenario, as
shown in Fig. 22(a). Compared to the other cases, the
PR-RPAD model with weights α1, α2, and α3 that take
the values of (1/2, 1/2, 0), (1/2, 0, 1/2), (0, 1/2, 1/2), and
(1/3, 1/3, 1/3) has a greater unintertwining accuracy curve
in Fig. 22(b). Thus, the deinterleaving performance of the
PR-RPAD model can be enhanced by reconstructing and
fusing the characteristics of the various pyramidal layers.
4) In open-set scenarios, the effect of CS generation
training on the PR-RPAD model’s RSD performance
is examined. Fig. 23 displays the PR-RPAD model
training loss versus accuracy curves for Experiments
4–6. In PR-RPAD, the CS generation training mod-
ule is not introduced during model training, while
in PR-RPAD+CS, the CS generation training mod-
ule is introduced during model training. The ﬁgure
shows that during the training of Experiments 4 and
Fig. 23.
Experiment 4–6: Training loss versus accuracy curve for the
PR-RPAD model.
5, the loss and accuracy curves of PR-RPAD+CS
and PR-RPAD overlap, making it challenging to
determine how much the CS generation training
module contributed to the RSD performance. The
PR-RPAD+CS loss curves in Experiment 6 con-
verge more quickly than PR-RPAD and are stable
around 40–60 iterations when the number of radar
radiation sources increases and the radiation pa-
rameters become more complex. In the meantime,
the PR-RPAD+CS RSD performance curve outper-
forms PR-RPAD. To sum up, it can be said that as
the number of radar radiation sources and radiation
parameter complexity increase, the CS generation
training module’s contribution to improving RSD
performance becomes more noticeable. This is par-
ticularly evident in Experiment 6.
5) To further demonstrate the superiority of the PR-
RPAD method proposed in this article, an ablation
study is conducted to verify the function and beneﬁts
of the algorithm’s Laplacian pyramid feature recon-
struction and fusion module in the deinterleaving of
radar signals. Using the data from Experiments 3 and
6, ablation experiments are designed. The Laplacian
PR module’s inclusion or absence in the method is
controlled to investigate the impact on the RSD per-
formance. Among these, reciprocal point adversarial
deinterleaving (RPAD) is the algorithm that does not
need the PR module. In the ablation experiment, the
model’s deinterleaving performance on the mixed
11250
IEEE TRANSACTIONS ON AEROSPACE AND ELECTRONIC SYSTEMS
VOL. 61, NO. 5
OCTOBER 2025
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 18]

TABLE XIII
Analysis of the Model’s Performance in the Ablation Studies via Mixed
Datasets for Experiments 3 (Closed Set) and 6 (Open Set)
TABLE XIV
Analysis of the Model’s Performance in the DTOA Control Experiments
via Mixed Datasets for Experiments 3 (Closed Set) and 6 (Open Set)
datasets of Experiments 3 and 6 is displayed in
Table XIII. The table illustrates that the application
of the Laplacian pyramid feature reconstruction and
fusion architecture can achieve tower reshaping of
the PDG features from low to high resolution, ex-
tend the different resolution characterization of the
PDG edges and texture features, and improve the
deinterleaving performance of radar-radiated source
signals in both the closed-set scenario (Experiment
3) and the open-set scenario (Experiment 6).
6) The PR-RPAD algorithm’s performance analysis
for deinterleaving when DTOA data is unavailable
or missing. Taking into account RSD’s real-time
requirements, the internal storage capacity limita-
tions following PDG synthesis, and the potential
for adverse effects from nearby pulses, a PDG is
generated from a single PDW when lenwindow = 1
is chosen. Only the TOA difference between neigh-
boring pulses is used in the transformation procedure
mentioned earlier; the temporal correlations between
other pulses are not investigated. Consequently, it
is possible to adjust whether or not the PDG has
DTOA information to conﬁrm its impact on the dein-
terleaving performance of radar signals; the PDG
that does not have DTOA information is known as
PDG-DTOA. When PDG-DTOA and PDG are input,
respectively, the model deinterleaving performance
results under the mixed dataset for Experiments 3
and 6 are displayed in Table XIV. The table shows
that, in both the closed-set scenario (Experiment 3)
and the open-set scenario (Experiment 6), DTOA
somewhat enhances the radar-radiated source sig-
nal deinterleaving performance. In addition, when
DTOA data are absent or unavailable, the PR-RPAD
method may nevertheless guarantee the regular op-
eration of the deinterleaved process, unlike the con-
ventional approach that solely uses TOA information
for deinterleaving.
VI.
CONCLUSION
In this study, we propose an open-set deinterleaving
technique for radar radiation source signals, called PR-
RPAD, which combines reciprocal point adversarial learn-
ing with Laplacian PR. The PR-RPAD algorithm employs
the visual characterization and feature enhancement of
the PDW to the PDG through the symmetric mapping
of grayscale matrices and DCCI_Otsu image enlargement
algorithms. Employing the Laplacian pyramid model of
multiresolution feature reconstruction and fusion to tower
reshape PDG features from low to high resolution. In paral-
lel, the concept of reciprocal points is introduced to model
the open space, and an instantiated enhancement method is
proposed. The deinterleaving of the known and unknown
radar radiation sources in the open space is ﬁnished under
the counter mechanism of reciprocal points and the known
radar. The following conclusions can be made in light of the
studies mentioned earlier.
1) This article has examined the deinterleaving of
signals from known and unknown radar radiation
sources using single-function, multifunction, and the
same type, as well as radar sources using common
PRI modulation styles (such as jitter PRI).
2) Unlike the conventional RSD approaches [5], [14],
[15], the PR-RPAD algorithm can handle complex
PRI modulation styles and searches for sequences
without determining the PRI or PRF. Many search
and merging rounds are not necessary for radar ra-
diation sources that have many pulses in a cycle.
3) The PR-RPAD algorithm can deinterleave signals
from multiple radar radiation sources using a single
network, as opposed to the RSD methods of neural
networks and automata [4], [16], [17], [18]. This
eliminates the need to train a different network for
each radar radiation source target and iteratively
iterates inputs and outputs to the radar source data.
4) In comparison to the SDIF, PRI-Tran, and SSD
(BLSTM, BGRU, and DCN) [19] algorithms, the
PR-RPAD algorithm is simple to train and converge,
and it exhibits notable performance advantages, ro-
bustness, and strong environmental adaptability for
known and unknown radar radiation source deinter-
leaving.
5) When pulse parameters are not complete, the PR-
RPAD method nevertheless performs reasonably
well in terms of deinterleaving. Nevertheless, PDG
creation, ampliﬁcation, and enhancement processing
LI ET AL.: RADAR SIGNAL OPEN-SET DEINTERLEAVING METHOD
11251
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 19]

result in higher algorithmic complexity and deinter-
leaving time cost, indicating the direction for future
algorithm optimization.
REFERENCES
[1] F. A. Butt and M. Jalil, “An overview of electronic warfare in radar
systems,” in Proc. Int. Conf. Technol. Adv. Elect., Electron. Comput.
Eng., Konya, Turkey, 2013, pp. 213–217.
[2] P. W. East and J. L. Everett, “Electronic support measures,” Proc.
Inst. Elect. Eng. F—Commun., Radar Signal Process., vol. 132, no. 4,
pp. 205–205, Jul. 1985.
[3] R. O. Schmidt, “On separating interleaved pulse trains,” IEEE
Trans. Aerosp. Electron. Syst., vol. AES-10, no. 1, pp. 162–166,
Jan. 1974.
[4] Z.-M. Liu and P. S. Yu, “Classiﬁcation, denoising, and deinter-
leaving of pulse streams with recurrent neural networks,” IEEE
Trans. Aerosp. Electron. Syst., vol. 55, no. 4, pp. 1624–1639,
Aug. 2019.
[5] Z. Ge, X. Sun, W. Ren, W. Chen, and G. Xu, “Improved algorithm
of radar pulse repetition interval deinterleaving based on pulse cor-
relation,” IEEE Access, vol. 7, pp. 30126–30134, 2019.
[6] Y. Liu and Q. Zhang, “Improved method for deinterleaving radar
signals and estimating PRI values,” IET Radar, Sonar, Navigat.,
vol. 12, no. 5, pp. 506–514, Mar. 2018.
[7] Y.Xi,X.Wu,Y.Wu,andL.Deng,“Afastandreal-timePRItransform
algorithm for deinterleaving large PRI jitter signals,” in Proc. 37th
Chin. Control Conf., Wuhan, China, 2018, pp. 4465–4469.
[8] S. Yang et al., “Deep contrastive clustering for signal deinterleaving,”
IEEE Trans. Aerosp. Electron. Syst., vol. 60, no. 1, pp. 252–263, Feb.
2024.
[9] W. Cheng, Q. Zhang, J. Dong, C. Wang, X. Liu, and G. Fang, “An
enhanced algorithm for deinterleaving mixed radar signals,” IEEE
Trans. Aerosp. Electron. Syst., vol. 57, no. 6, pp. 3927–3940, Dec.
2021.
[10] Q. Guo, S. Huang, L. Qi, D. Li, and M. Kaliuzhnyi, “A radar
signal deinterleaving method based on complex network and
Laplacian graph clustering,” IEEE Signal Process. Lett., vol. 31,
pp. 2580–2584, Sep. 2024.
[11] Y. Zhou, Y. Zheng, S. Wei, L. Zhang, and Z. Wen, “CAU-net: A
convolutional attention U-network for radar signal deinterleaving,”
IEEE Commun. Lett., vol. 28, no. 7, pp. 1569–1573, Jul. 2024.
[12] Q. Zhu and C. Shang, “Survey of signal parameter identiﬁcation
technology,” in Proc. Int. Conf. Comput. Syst., Electron. Control,
Dalian, China, 2017, pp. 714–716.
[13] S. Jinping, L. Zhen, L. Li, and L. Xiang, “Progress in radar emitter
signal deinterleaving,” J. Radars, vol. 11, no. 3, pp. 418–433, Jun.
2022.
[14] J. Liu, H. Meng, Y. Liu, and X. Wang, “Deinterleaving pulse
trains in unconventional circumstances using multiple hypothesis
tracking algorithm,” Signal Process, vol. 90, no. 8, pp. 2581–2593,
Aug. 2010.
[15] N. Visnevski, S. Haykin, V. Krishnamurthy, F. A. Dilkes, and
P. Lavoie, “Hidden Markov models for radar pulse train analysis in
electronic warfare,” in Proc. IEEE Int. Conf. Acoust., Speech, Signal
Process., Philadelphia, PA, USA, 2005, pp. v/597–v/600.
[16] X. Li, Z.-M. Liu, and Z. Huang, “Denoising of radar pulse
streams with autoencoders,” IEEE Commun. Lett., vol. 24, no. 4,
pp. 797–801, Apr. 2020.
[17] X. Li, Z. Liu, and Z. Huang, “Deinterleaving of pulse streams
with denoising autoencoders,” IEEE Trans. Aerosp. Electron. Syst.,
vol. 56, no. 6, pp. 4767–4778, Dec. 2020.
[18] Z.-M. Liu, “Online pulse deinterleaving with ﬁnite automata,” IEEE
Trans. Aerosp. Electron. Syst., vol. 56, no. 2, pp. 1139–1147, Apr.
2020.
[19] W. Chao, S. Liting, L. Zhangmeng, and H. Zhitao, “A radar signal
deinterleaving method based on semantic segmentation with neural
network,” IEEE Trans. Signal Process., vol. 70, pp. 5806–5821,
2022.
[20] S. Vakin, L. Shustov, and R. Dunwell, Fundamentals of Electronic
Warfare. Norwood, MA, USA: Artech House, 2001, pp. 13–14.
[21] K. Krishna and M. N. Murty, “Genetic K-means algorithm,” IEEE
Trans. Syst., Man, Cybern. B, Cybern., vol. 29, no. 3, pp. 433–439,
Jun. 1999.
[22] M. G. S. Ahmed and B. Tang, “Sorting radar signal from symme-
try clustering perspective,” J. Syst. Eng. Electron., vol. 28, no. 4,
pp. 690–696, Aug. 2017.
[23] Z. Yu, Y. Wang, and C. Chen, “Radar emitter signal sorting method
based on density clustering algorithm of signal aliasing degree judg-
ment,” in Proc. 15th IEEE Conf. Ind. Electron. Appl., Kristiansand,
Norway, 2020, pp. 1027–1031.
[24] H. Mardia, “New techniques for the deinterleaving of repetitive se-
quences,” Proc. Inst. Elect. Eng. F—Radar Signal Process., vol. 136,
no. 4, pp. 149–154, Aug. 1989.
[25] D. Milojevic and B. Popovic, “Improved algorithm for the deinter-
leaving of radar pulses,” Proc. Inst. Elect. Eng. F—Radar Signal
Process., vol. 139, no. 1, pp. 98–104, Feb. 1992.
[26] K. Nishiguchi and M. Kobayashi, “Improved algorithm for estimat-
ing pulse repetition intervals,” IEEE Trans. Aerosp. Electron. Syst.,
vol. 36, no. 2, pp. 407–421, Apr. 2000.
[27] S. Yuan and Z.-M. Liu, “Temporal feature learning and pulse pre-
diction for radars with variable parameters,” Remote Sens., vol. 14,
no. 21, Oct. 2022, Art. no. 5439.
[28] L. Cheng, A. Yang, and Z. Liu, “A radar main lobe pulse correlation
sorting method,” in Proc. IEEE 4th Int. Conf. Signal Image Process.,
Wuxi, China, 2019, pp. 41–45.
[29] D. Xu, M. Xu, H. Wang, F. Feng, L. Tang, and M. Gu, “A real-time
radar signal sorting method and implementation based on DSP,”
in Proc. IEEE MTT-S Int. Wirel. Symp., Shanghai, China, 2020,
pp. 1–3.
[30] S. Yuan, S.-Q. Kang, W.-X. Shang, and Z.-M. Liu, “Reconstruction
of radar pulse repetition pattern via semantic coding of intercepted
pulse trains,” IEEE Trans. Aerosp. Electron. Syst., vol. 59, no. 1,
pp. 394–403, Feb. 2023.
[31] G. Seroussi, W. Szpankowski, and M. J. Weinberger, “Deinterleaving
ﬁnite memory processes via penalized maximum likelihood,” Aug.
2011, arXiv: 1108.5212.
[32] T. Conroy and J. B. Moore, “The limits of extended Kalman ﬁltering
for pulse train deinterleaving,” IEEE Trans. Signal Process., vol. 46,
no. 12, pp. 3326–3332, Dec. 1998.
[33] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, “A density-based
algorithm for discovering clusters in large spatial databases with
noise,” in Proc. 2nd Int. Conf. Knowl. Discov. Data Mining, Portland,
OR, USA, 1996, pp. 226–231.
[34] R. S. Kumaran, “Ordering points to identify the clustering structure
(optics) with ant colony optimization for wireless sensor networks,”
Eur. J. Sci. Res., vol. 59, no. 4, pp. 571–582, Feb. 2013.
[35] L. Kaufman and P. J. Rousseeuw, Agglomerative Nesting (Program
AGNES). Hoboken, NJ, USA: Wiley, 2008, pp. 199–252.
[36] A. Ben-Hur, D. Horn, H. T. Siegelmann, and V. Vapnik, “Support
vector clustering,” J. Mach. Learn. Res., vol. 2, pp. 125–137, Mar.
2002.
[37] S. Su, X. Fu, C. Zhao, J. Yang, M. Xie, and Z. Gao, “Unsupervised
k-means combined with SOFM structure adaptive radar signal sort-
ing algorithm,” in Proc. IEEE Int. Conf. Signal, Inf. Data Process.,
Chongqing, China, 2019, pp. 1–4.
[38] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks
forsemanticsegmentation,”inProc.IEEEConf.Comput.Vis.Pattern
Recognit., Boston, MA, USA, 2015, pp. 3431–3440.
[39] W. J. Scheirer, A. de Rezende Rocha, A. Sapkota, and T. E. Boult,
“Toward open set recognition,” IEEE Trans. Pattern Anal. Mach.
Intell., vol. 35, no. 7, pp. 1757–1772, Jul. 2013.
[40] W. J. Scheirer, L. P. Jain, and T. E. Boult, “Probability models
for open set recognition,” IEEE Trans. Pattern Anal. Mach. Intell.,
vol. 36, no. 11, pp. 2317–2324, Nov. 2014.
[41] E. M. Rudd, L. P. Jain, and W. J. Scheirer, “The extreme value
machine,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 40, no. 3,
pp. 762–768, Mar. 2018.
11252
IEEE TRANSACTIONS ON AEROSPACE AND ELECTRONIC SYSTEMS
VOL. 61, NO. 5
OCTOBER 2025
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.


[page 20]

[42] P. R. M. Junior et al., “Nearest neighbors distance ratio open-set
classiﬁer,” Mach. Learn., vol. 106, no. 3, pp. 359–386, Mar. 2017.
[43] H. Zhang and V. M. Patel, “Sparse representation-based open set
recognition,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 39, no. 8,
pp. 1690–1696, Aug. 2017.
[44] A. Bendale and T. E. Boult, “Towards open set deep networks,” in
Proc. IEEE Conf. Comput. Vis. Pattern Recognit., Las Vegas, NV,
USA, 2016, pp. 1563–1572.
[45] L. Shu, H. Xu, and B. Liu, “DOC: Deep open classiﬁcation of text
documents,” Sep. 2017, arXiv: 1709.08716.
[46] M. Krichen, “Generative adversarial networks,” in Proc. 14th Int.
Conf. Comput. Commun. Netw. Technol., New Delhi, India, 2023,
pp. 1–7.
[47] X. Sun, Z. Yang, C. Zhang, K.-V. Ling, and G. Peng, “Conditional
Gaussian distribution learning for open set recognition,” in Proc.
IEEE/CVF Conf. Comput. Vis. Pattern Recognit., Seattle, WA, USA,
2020, pp. 13477–13486.
[48] H. Zhang, A. Li, J. Guo, and Y. Guo, “Hybrid models for open set
recognition,” in Proc. Eur. Conf. Comput. Vis., Glasgow, U.K., 2020,
pp. 102–117.
[49] S. M. E. B. Harb, N. A. M. Isa, and S. A. Salamah, “Improved image
magniﬁcation algorithm based on Otsu thresholding,” Comput. Elect.
Eng., vol. 46, no. 1, pp. 338–355, Aug. 2015.
[50] S. Bai, J. Z. Kolter, and V. Koltun, “An empirical evaluation of
generic convolutional and recurrent networks for sequence model-
ing,” 2018, arXiv: 1803.01271.
[51] A. R. Dhamija, M. Günther, and T. Boult, “Reducing network ag-
nostophobia,” in Proc. 32nd Annu. Conf. Neural Inf. Process. Syst.,
Montréal, QC, Canada, 2018, pp. 9157–9168.
Wenbo Li received the B.Eng. degree in
information countermeasure technology from
Shenyang Ligong University, Shenyang, China,
in 2020. He is currently working toward the
Ph.D. degree in electronic science and technol-
ogy with Xidian University, Xi’an, China.
His current research interests include deep
learning, radar emitter deinterleaving, and clas-
siﬁcation.
Yang-Yang Dong (Member, IEEE) received the
B.Eng. and Ph.D. degrees in electronic science
and technology from Xidian University, Xi’an,
China, in 2012 and 2017, respectively.
Since 2017, he has been a Lecturer and then
an Associate Professor with the School of Elec-
tronic Engineering, Xidian University. His re-
search interests include array signal processing
and intelligent spectrum sensing.
Chunxi Dong was born in Sanmenxia, Henan,
China.HereceivedthePh.D.degreeinelectronic
science and technology from Xidian University,
Xi’an, China, in 2004.
He is currently a Professor and the Dean of the
Department of Information Technology, School
of Electronic Engineering, Xidian University.
His research interests include high-speed signal
processing and system simulation.
Ronghua Guo received the B.Eng. degree in
information countermeasure technology in 2020
from Xidian University, Xi’an, China, where she
is currently working toward the Ph.D. degree in
electronic science and technology.
Her current research interests include pulse-
splitting detection and radar emitter deinterleav-
ing.
Zhiyuan Li received the B.Eng. degree in com-
munication engineering from Dalian University,
Dalian, China, in 2020. He is currently working
toward the Ph.D. degree in electronic science
and technology with Xidian University, Xi’an,
China.
His current research interests include deep
learning and radar mode identiﬁcation.
LI ET AL.: RADAR SIGNAL OPEN-SET DEINTERLEAVING METHOD
11253
Authorized licensed use limited to: JILIN UNIVERSITY. Downloaded on March 23,2026 at 14:14:02 UTC from IEEE Xplore.  Restrictions apply.
