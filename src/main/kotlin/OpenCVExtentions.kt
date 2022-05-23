import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

private fun Mat.runOperation(operation: (Mat, Mat) -> Unit): Mat {
    val dst = this.clone()
    operation(this, dst)
    return dst
}

fun Mat.threshold(thresh: Double = 0.0, maxVal: Double = 255.0, type: Int) = this.runOperation { src, dst ->
    Imgproc.threshold(src, dst, thresh, maxVal, type)
}

fun Mat.dilate(kernel: Mat) = this.runOperation { src, dst ->
    Imgproc.dilate(src, dst, kernel)
}

fun Mat.erode(kernel: Mat) = this.runOperation { src, dst ->
    Imgproc.erode(src, dst, kernel)
}

operator fun Mat.minus(mat: Mat) = this.runOperation { src, dst ->
    Core.subtract(src, mat, dst)
}

fun Mat.grayscale() = this.runOperation { src, dst ->
    Imgproc.cvtColor(src, dst, Imgproc.COLOR_BGR2GRAY)
}

fun Mat.getContours(): List<MatOfPoint> {
    val contours = mutableListOf<MatOfPoint>()
    Imgproc.findContours(this, contours, Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, Point(0.0, 0.0))
    return contours
}

fun Mat.adaptiveHistogramEqualization() = this.runOperation { src, dst ->
    Imgproc.createCLAHE(2.0, Size(5.0, 5.0)).apply(src.grayscale(), dst)
}

fun Mat.drawRectangles(
    rectangles: List<Rect>,
    color: Scalar = Scalar(0.0, 0.0, 255.0),
    thickness: Int = 2
) = this.runOperation { _, dst ->
    rectangles.forEach { Imgproc.rectangle(dst, it, color, thickness) }
}