import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc


fun Mat.preprocess() = this.adaptiveHistogramEqualization()

fun Mat.getSkeleton() = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(3.0, 3.0)).let {
    this.dilate(it) - this.erode(it)
}

fun Mat.thresholdOTSU(): Mat = this.threshold(type = Imgproc.THRESH_BINARY or Imgproc.THRESH_OTSU)

fun Mat.morphClose() = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0)).let {
    this.dilate(it).erode(it)
}

fun Mat.getTextRectangles() = this.getContours().map { it to Imgproc.boundingRect(it) }.filter { (contour, rect) ->
    (rect.width.toDouble() / rect.height < 20.0 && Imgproc.contourArea(contour) / rect.area() > 0.25 && rect.area() / (this.height() * this.width()) > 0.005)
}.map { it.second }

fun Mat.localizeText(src: Mat) = src.drawRectangles(this.getTextRectangles())

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        System.err.println("Enter the path to image followed by the command")
        return
    }
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    val image = Imgcodecs.imread(args[0])
    val localized = image.preprocess().getSkeleton().thresholdOTSU().morphClose().localizeText(image)

    Imgcodecs.imwrite("${args[0].substring(0, args[0].lastIndexOf('.'))}_localized.jpg", localized)
}