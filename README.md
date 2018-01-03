# StereoMatchingAlgorithm
-Description:

issued from the articles "global solutions of variationnal models woth convex regularisation" and "http://www.numdam.org/article/SMAI-JCM_2015__1__29_0.pdf" (the link to the report that summarizes the internship to be joined later)

-Installation:

You need to install the version 3.3.1 of opencv ( check also that the older versions of opencv are removed)

https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html

Add the following path to the profile script:
export PKG_CONFIG_PATH=/usr/local/include/opencv3/lib/pkgconfig
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/include/opencv3/lib
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib/



Usage:
once the executable is built, you should launch the algorithm with the following options:

-im1: the first image from which the disprity is computed
-im2: the target image, the disparity is computed from the first image to the second one.
-dataTerm: used data-term for the computation of the disparity, the dataTerm is a 3D matrix expressed in the form of a 3D matrix, with a term f the form g(i,j,k)=cost(im1(i,j),im2(i,j+K)), where cost can be choosen to be equal to the absolute difference by choosing 'absdiff', or the census signature by choosing 'census'.
You can implement new data terms, in the class "MatchingAlgorithm".

-tsize: is the length of the interval of disparity.
-zoom: is the inverse of the distance between two adjacent disparities, if it is bigger than 1, we will have a finer disparity map, otherwise the disparity map will be smaller.

-path_to_disparity: specify the path where the disparity will be written (the image is overwritten !)
-multiscale: it is a multiscale approach for which the primal variables of the 3D ROF algorithm are iterated for a smaller size, then resized to be used in the 3DROF algorithm, again, but this approach doesn't seem to be very efficient.

-Niter: the number of maximal iterations before the algorithm would stop.

-ratioGap: the ratio between the primalDualGap at the first iteration and at the current iteration for which the algorithm would stop, the initialDualGap is set to be equal to 0 before the first iteration.

An example of use would be the following:

./exec --im1 image1_gray_Art.tif --im2 image2_gray_Art.tif  --dataterm census --tsize 20 --offset 0 --path_to_disparity disparityGrayArt_census.tif --threadsMax 32 --method accelerated --Niter 20 --ratioGap 0.0000000001 --zoom 1
