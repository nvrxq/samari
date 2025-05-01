<div align="center">
<img align="left" width="100" height="100" src="./assets/samari_logo.png" alt="">

### <span style="color: blue;">SAMARI</span> : SAM2 Markov Chain Filter
[Gleb Kirichenko](https://github.com/nvrxq), [Dmitriy Yurtov](https://github.com/Karniton) 
</div>

<div align="center">

## 🖥 Demo

<video width="1300" height="360" controls>
  <source src="https://drive.google.com/file/d/1QET9HYezDVNEE8zuGYJckZu1LqMdCV84/view?usp=sharing" type="video/mp4">
</video>
</div>


<div align="center">

## 📌 Project Info

</div>

**The final work of the 4th year of the HSE Bachelor's degree. Stable object tracker on video.We present a Zero-Shot filtering method for the SAM 2 model.** 



**Links to contact the authors** :
- [Gleb Kirichenko](t.me/nvrxq)
- [Dmitriy Yurtov](t.me/dima11628)
<div align="center">

##  🔎 Experiments. (What worked, what didn't)
</div>

-  [Dmitriy Yurtov](t.me/dima11628) : In the first iterations of the project, we wanted to speed up the Kalman filter, this was done to improve this filter not only for computer vision tasks. The Kalman filter code on the GPU is in `samari/experiments`. [✅]

- [Gleb Kirichenko](t.me/nvrxq): To improve the quality, inspired by the [MCMCDA](https://engineering.ucmerced.edu/sites/engineering.ucmerced.edu/files/page/documents/2008techreport-oh.pdf) article, we made a filter for the SAM 2 model based on this filter. The MCMCDA filter code is in `sam2/sam2/modeling/mcmcda_filter.py`[✅]
- [Dmitriy Yurtov](t.me/dima11628): To speed up the MCMCDA tracker, we have also added several options such as :
    - *num_mcmc_iterations* - **number of iterations for MCMC.**   [<span style="color: green;">Default = 1000</span>][✅]
    - *update_freq* - **track refresh rate (every N frames)** [<span style="color: green;">Default = 4</span>][✅]
- [Gleb Kirichenko](t.me/nvrxq):A learnable Kalman filter, using various architectures. Either `boxes` or `images, boxes` is accepted as input.
<div align="center">

##  ⚙️ Setup
</div>
<div align="center">
</div>

**In order to test the filter along with SAM2 on your video, you need to take a few steps.**
- **Downloading models**:
    `sh download_ckpt.sh`
- **Install SAM2**:
```
cd sam2
pip install -e .
pip install -e ".[notebooks]"
```
- **Run demo**
```
python3 run_demo.py --video-input /PATH/TO/VIDEO --video_output /PATH/TO/SAVE --label-file
```
- **Create Side-by-Side**
```
python3 create_sbs_video.py --sam2-result /PATH/TO/SAM2VIDEO --exp-result /PATH/TO/YOUR/EXP -exp-name
```
<div align="center">

##  📊 Results
</div>

### Speed & Throughput
#### 🛠 Setup:
- **GPU: 1xA100(80GB)** 
- **Model: SAM-2 Base & default params for MCMCDA**
- **Result** : **14** `Frames/Second`
### 🖊 Quality:
- **To be soon......**