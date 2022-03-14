# Coral
Using image recognition to determine the health of a coral colony by comparing its current condition to past data

Please check the png image files for a visualising explanation

Task:
A large coral colony will be located in the reef. The coral colony will be constructed from ½-inch PVC pipe. The base of the coral will be located in two squares in the coral reef grid. Companies must determine the health of the coral colony by comparing the current image to an image taken one year ago. Companies must compare images to assess whether the coral has grown, is damaged/has died, is bleached/blotched, or has recovered from bleaching/blotching. Growth is defined as the coral colony having new branches not seen in the previous image. Damage or death is defined as the coral colony missing branches that were seen in the previous image. Coral bleaching or blotching is defined as colored branches from the previous photo having turned white. Recovery from bleaching or blotching is defined as white branches from the previous photo now being colored. 
Companies must compare the current coral colony to the previous image and show any changes on a video display. The base of the coral will be painted black, will not change shape or color, and should not be used in the image comparisons. 
The following parameters must be used:
Areas of growth should be outlined with a green overlay or a marked with a green rectangle/circle around the affected area.
Areas of damage or death should be outlined with a yellow overlay or marked with a yellow rectangle/circle around the affected area.
Areas of bleaching/blotching should be outlined with a red overlay or marked with a red rectangle/circle around the affected area.
Areas that have recovered from bleaching/blotching should be outlined with a blue overlay or marked with a blue rectangle/circle around the affected area.

Solution:
The coral colony picture is masked using the pink and white colour ranges to produce a mask for each colour as well as a combined mask of the whole coral colony. The HSV colour ranges of the after image are found using the KMeans clustering algorithm, where n=3 is passed through as the parameter for the number of clusters to be found, which includes the pink, white and background colour.  
Both masks of the before and after corals are being fed to the ORB detector, and the key points are found. The key points are then being matched using brute force matching, and only the top 90% of matches are being used to reduce inaccuracies. Homography is found using the matched points and the perspective of the before mask is fixed using homography to match the mask of the after coral colony. Then the masks are being compared bitwise to obtain the changes of the colony.
