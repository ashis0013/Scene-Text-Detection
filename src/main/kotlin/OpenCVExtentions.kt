import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

private fun Mat.runOperation(operation: (Mat, Mat) -> Unit): Mat {
    val dst = this.clone()
    operation(this, dst)
    return dst
}

fun Mat.threshold(thresh: Double = 0.0, maxVal: Double = 255.0, type: Int) = this.runOperation { src, dst ->
    Imgproc.threshold(this, dst, thresh, maxVal, type)
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