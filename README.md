# Coral  
Using image recognition to determine the health of a coral colony by comparing its current condition to past data  
  
Task:  
Compare images to assess whether the coral has grown, is damaged/has died, is bleached/blotched, or has recovered from bleaching/blotching. Growth is defined as the coral colony having new branches not seen in the previous image. Damage or death is defined as the coral colony missing branches that were seen in the previous image. Coral bleaching or blotching is defined as colored branches from the previous photo having turned white. Recovery from bleaching or blotching is defined as white branches from the previous photo now being colored.  
  
The following parameters are used:  
Green = Areas of growth  
Yellow = Areas of damage or death  
Red = Areas of bleaching/blotching  
Blue = Areas that have recovered from bleaching/blotching  
  
![alt text](https://github.com/teethoe/Coral/blob/main/sample%20test%20result.png?raw=true)  
(Note that the before and after images shown above have already been processed by the program (cropped and perspective fixed), they are not the orginal sample images fed to the program)  
  
Solution:  
The coral colony picture is masked using the pink and white colour ranges to produce a mask for each colour as well as a combined mask of the whole coral colony. The HSV colour ranges of the images are found using the KMeans clustering algorithm, where n=3 is passed through as the parameter for the number of clusters to be found, which includes the pink, white and background colour. (the "cluster" function in the Process class in https://github.com/teethoe/Coral/blob/main/process.py)  
Both masks of the before and after corals are being fed to the ORB detector, and the key points are found. The key points are then being matched using brute force matching, and only the top 90% of matches are being used to reduce inaccuracies. Homography is found using the matched points and the perspective of the before mask is fixed using homography to match the mask of the after coral colony. (the "fix" function in https://github.com/teethoe/Coral/blob/main/process.py)  
Then the masks are being compared bitwise to obtain the changes of the colony.   
  
    
![alt text](https://github.com/teethoe/Coral/blob/main/Coral%20Process%20Diagram.png?raw=true)  
![alt text](https://github.com/teethoe/Coral/blob/main/Coral%20Change%20Determination.png?raw=true)  
