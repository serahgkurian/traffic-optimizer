Image colorization transforms grayscale images into visually convincing color representa
tions by leveraging reference color images to guide the colorization process. This report
 presents an implementation and analysis of exemplar-based deep learning networks for
 reference-guided image colorization. The exemplar-based approach reduces user effort by
 transferring colors from reference images that visually resemble grayscale targets through
 semantic correspondence matching and neural network-based color prediction. We imple
ment a framework comprising semantic correspondence matching using pretrained VGG
19 features and U-Net-based decoder architecture for color reconstruction. The system is
 evaluated on the Imagenette dataset using quantitative metrics including PSNR, SSIM,
 MSE, and LPIPS. Our experimental results demonstrate that the exemplar-based ap
proach achieves superior performance with PSNR of 24.12 dB and SSIM of 0.9384, pro
ducing high-quality colorizations when appropriate reference images are available. The
 method shows particular strength in maintaining object-level color consistency through
 semantic correspondence matching while effectively transferring colors from visually sim
ilar reference images.
