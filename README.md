# SRNet
A powerful structure to compress and reconstruct CSI.
https://ieeexplore.ieee.org/document/9585309
## Usage
**Step1**

We train and evaluate the model with PCM method proposed in our paper. Hence, before training or evalution, we have to convert the data from original csi matrix to the format introduced by PCM method. 

The operation is written in matlab files and stored in "generate_pcm_data" folder. Before running matlab codes, you should modify the path for loading input data. And the original csi data is referred in https://github.com/sydney222/Python_CsiNet .

Noted that the above conversion can also be implemented in Python.

**Step2**
```
# Assuming the compression ratio is 4, the scenario is indoor, and the data path is ../data
python main.py --cr 4 --scenario indoor --data-root ../data
```
## Evalution
The performance of P-SRNet under different compression ratio and scenario.  
Download checkpoint files from the "checkpoints" folder or from google drive https://drive.google.com/drive/folders/1jkAaRtKjffhCkyajUI7F42xL-AzoAkp-?usp=sharing
<table>
    <tr>
        <td>CR</td> 
        <td>scenario</td> 
        <td>NMSE(dB)</td>
        <td>trained model</td>
   </tr>
    <tr>
        <td rowspan="2">4</td>    
        <td >indoor</td>  
        <td >-24.23</td>
        <td ><a href="https://github.com/MoliaChen/SRNet/blob/main/checkpoints/indoor_4.pth">indoor cr4</a></td>
    </tr>
    <tr>
        <td >outdoor</td>
        <td >-15.43</td>
        <td ><a href="https://github.com/MoliaChen/SRNet/blob/main/checkpoints/outdoor_4.pth">outdoor cr4</a></td>
    </tr>
    <tr>
          <td rowspan="2">8</td>    
          <td >indoor</td>  
          <td >-19.26</td>  
          <td ><a href="https://github.com/MoliaChen/SRNet/blob/main/checkpoints/indoor_8.pth">indoor cr8</a></td>
      </tr>
      <tr>
          <td >outdoor</td>
          <td >-13.47</td>
          <td ><a href="https://github.com/MoliaChen/SRNet/blob/main/checkpoints/outdoor_8.pth">outdoor cr8</a></td>
      </tr>
    <tr>
          <td rowspan="2">16</td>    
          <td >indoor</td>  
          <td >-15.26</td>  
           <td ><a href="https://github.com/MoliaChen/SRNet/blob/main/checkpoints/indoor_16.pth">indoor cr16</a></td>
      </tr>
      <tr>
          <td >outdoor</td>
          <td >-11.31</td>
          <td ><a href="https://github.com/MoliaChen/SRNet/blob/main/checkpoints/outdoor_16.pth">outdoor cr16</a></td>
      </tr>
      <tr>
          <td rowspan="2">32</td>    
          <td >indoor</td>  
          <td >-11.61</td>  
          <td ><a href="https://github.com/MoliaChen/SRNet/blob/main/checkpoints/indoor_32.pth">indoor cr32</a></td>
      </tr>
      <tr>
          <td >outdoor</td>
          <td >-9.17</td>
          <td ><a href="https://github.com/MoliaChen/SRNet/blob/main/checkpoints/outdoor_32.pth">outdoor cr32</a></td>
      </tr>
     <tr>
          <td rowspan="2">64</td>    
          <td >indoor</td>  
          <td >-8.27</td> 
         <td ><a href="https://github.com/MoliaChen/SRNet/blob/main/checkpoints/indoor_64.pth">indoor cr64</a></td>
      </tr>
      <tr>
          <td >outdoor</td>
          <td >-7.80</td>
          <td ><a href="https://github.com/MoliaChen/SRNet/blob/main/checkpoints/outdoor_64.pth">outdoor cr64</a></td>
      </tr>
</table>
