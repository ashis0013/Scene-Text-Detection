# Scene Text Detection With Gradient Morphology

Detecting scene text (Text in wild images not in noob documents) with basic image processing techniques like image-dilation and erosion and absolutely zero Machine Learning 😲 
This is built with OpenCV only because the image processing methods are readily available. To run this use `detext.jar` inside `jar-build` folder. Running it will create a 'filename_localized.jpg' in the same path as of the input.

```shell
java -jar jar-build/detextscene.jar test.jpeg
```

Note: You need Java 18 to run this as we are using very new version of OpenCV

Read the paper [here](https://www.igi-global.com/article/multilingual-scene-text-detection-using-gradient-morphology/258252)

Cite us at
```bibtex
@article{dhar2020multilingual,
  title={Multilingual scene text detection using gradient morphology},
  author={Dhar, Dibyajyoti and Chakraborty, Neelotpal and Choudhury, Sayan and Paul, Ashis and Mollah, Ayatullah Faruk and Basu, Subhadip and Sarkar, Ram},
  journal={International Journal of Computer Vision and Image Processing (IJCVIP)},
  volume={10},
  number={3},
  pages={31--43},
  year={2020},
  publisher={IGI Global}
}
```
