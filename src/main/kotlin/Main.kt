import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

fun runMorphologicalOperators(input:Mat): Mat {
    val dilated = input.clone()
    val eroded = input.clone()
    val subtracted = input.clone()
    val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0,3.0))
    Imgproc.dilate(input, dilated, kernel)
    Imgproc.erode(input, eroded, kernel)
    Core.subtract(dilated, eroded, subtracted)
    return subtracted
}

fun main(args: Array<String>) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val image = Imgcodecs.imread("/Users/ashis.paul/Documents/scene-text-detection/res/download.jpeg")
    Imgcodecs.imwrite("/Users/ashis.paul/Documents/scene-text-detection/res/morphed.jpg", runMorphologicalOperators(image))
}