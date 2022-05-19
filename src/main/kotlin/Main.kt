import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc


fun Mat.preprocess() = this.grayscale()

fun Mat.getSkeleton() = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0,3.0)).let {
    this.dilate(it) - this.erode(it)
}

fun Mat.thresholdOTSU(): Mat = this.threshold(type = Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)

fun Mat.morphClose() = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0,3.0)).let {
    this.dilate(it).erode(it)
}

fun main(args: Array<String>) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val image = Imgcodecs.imread("/Users/ashis.paul/Documents/scene-text-detection/res/download.jpeg")
    val final = image.preprocess().getSkeleton().thresholdOTSU().morphClose()
    Imgcodecs.imwrite("/Users/ashis.paul/Documents/scene-text-detection/res/morphed.jpg", final)
}