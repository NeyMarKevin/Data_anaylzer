# Logger plotter for GL900 
***
**This code is only for reading the .csv file in mini logger GL900, it cannot handle the .gbd file, the .gbd file needs GL900 APS to convert to .csv file first.**
***
For using this plotter, running the GUI.py and dragging the file in the window.
***
## Important tips: 

- There are always a lot of data in the .csv file, to make sure the figure is readable, this code uses a **skip_step** to decrease the number of points. The default volume is 1,000, which means it will pick one point from each 1,000 points to generate the graph, this value can be changed in the **read_scanning_file.py**, the definition is at the start of the coding

- The **pre_process** function are using the other 7 channels signal to decrease the impact the noise, however, it needs to be changed when the logger is conneted to the ground or more channel is connected. This can be changed in the **voltage = float(mainData[i][4]) + noise** line