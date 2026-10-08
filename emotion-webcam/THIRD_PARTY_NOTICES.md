# Third-party notices

The application source is MIT licensed (LICENSE). Python dependencies retain
their respective licenses in their installed package metadata. No dependency
binaries are redistributed in this project ZIP.

`tests/fixtures/face.png` is a grayscale transformation of the NASA portrait
of astronaut Eileen Collins distributed as `skimage.data.astronaut` in
scikit-image 0.25.2. The upstream documentation identifies the NASA image as
public domain. It is used only as a software-integration fixture, with no
ground-truth emotion label and no accuracy claim about the pictured person.

Source: https://scikit-image.org/docs/0.25.x/api/skimage.data.html#skimage.data.astronaut

Haar cascade XML files are supplied by the installed OpenCV package; its
license and detector notices remain in that package.
