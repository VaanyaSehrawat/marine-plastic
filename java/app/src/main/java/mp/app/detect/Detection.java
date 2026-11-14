package mp.app.detect;

/** One detected object. letterboxXyxy is in the 1280x1280 padded frame, imageXyxy in original image pixels. */
public record Detection(int cls, String name, float conf, float[] letterboxXyxy, float[] imageXyxy) {}
