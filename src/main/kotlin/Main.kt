import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc


fun Mat.preprocess(): Mat {
    val dst = this.clone()
    Imgproc.cvtColor(this, dst, Imgproc.COLOR_BGR2GRAY)
    return dst
}

fun Mat.dilateAndErode(isSubtracting: Boolean): Mat {
    val dilated = this.clone()
    val eroded = this.clone()
    val subtracted = this.clone()
    val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0,3.0))
    Imgproc.dilate(this, dilated, kernel)
    Imgproc.erode(if (!isSubtracting) dilated else this, eroded, kernel)
    if (isSubtracting) {
        Core.subtract(dilated, eroded, subtracted)
    }
    return if (isSubtracting) subtracted else eroded
}

fun Mat.getSkeleton() = this.dilateAndErode(isSubtracting = true)

fun Mat.thresholdOTSU(): Mat {
    val dst = this.clone()
    Imgproc.threshold(this, dst, 0.0, 255.0, Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)
    return dst
}

fun Mat.morphClose() = this.dilateAndErode(isSubtracting = false)

fun main(args: Array<String>) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val image = Imgcodecs.imread("/Users/ashis.paul/Documents/scene-text-detection/res/download.jpeg")
    val final = image.preprocess().getSkeleton().thresholdOTSU().morphClose()
    Imgcodecs.imwrite("/Users/ashis.paul/Documents/scene-text-detection/res/morphed.jpg", final)
}