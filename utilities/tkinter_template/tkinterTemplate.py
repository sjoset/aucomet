#!/usr/bin/env python3
"""
Title: A Tkinter_GUI_Template.py
Description: A Tkinter GUI Template to build basic graphically driven
interface.

@author: Nathan Davis
"""

# Tkinter Module for access to Widgets and other GUI Assessts
import tkinter as tk
# from tkinter import messagebox
# from tkinter.filedialog import askopenfilename

# Matplotlib Imports with defined TkAgg backend for embedding in Tkinter GUI
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# Tkinter Class Definition
class Application(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        # self.master.geometry("1080x1900")
        self.master.wm_title("Tkinter GUI App")

        # Initialize Application Attributes

        """--Methods to Construct Front Panel--"""

        # --> Make Matplotlib Display to View Fits File
        self.makeGraphPlot()

        # --> Function to Define Button on Application Frame
        self.makeButton()

        # -->Make Button Example to Plot Sample Data
        self.makePlotButton()

    def makePlotButton(self):
        # Example of how to plot data on plot on tkinter GUI
        def plotTestData():
            x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

            self.graphAxes.scatter(x, y)
            self.graphCanvas.draw()
        testPlot = tk.Button(self.master, text="Plot Data",
                             bg="grey", command=plotTestData)
        testPlot.place(relwidth=0.2,
                       relheight=0.1,
                       relx=0.0,
                       rely=0.11)

    def makeButton(self):
        def testButtonAction():
            print("Button Pressed...")
        testButton = tk.Button(self.master, text="Test Button",
                               bg="grey", command=testButtonAction)
        testButton.place(relwidth=0.2,
                         relheight=0.1,
                         relx=0.0,
                         rely=0.0)

    def makeGraphPlot(self):
        # Make Graph Frame to Draw Plot
        graphFrame_label = tk.Label(self.master, text="Plot")
        graphFrame_label.place(relwidth=0.05,
                               relheight=0.03,
                               relx=0.45,
                               rely=0.02)
        graphFrame = tk.Frame(self.master,
                              relief=tk.SUNKEN,
                              bd=1,
                              bg="gray")
        graphFrame.grid_propagate(False)
        graphFrame.place(relwidth=0.4,
                         relheight=0.4,
                         relx=0.45,
                         rely=0.05)
        # Setup Graph Figure, Axes, and Canvas to display in frame
        graphFig = Figure(figsize=(5, 5), dpi=100)
        self.graphAxes = graphFig.add_subplot(111)
        self.graphAxes.grid(True)
        self.graphCanvas = FigureCanvasTkAgg(graphFig, master=graphFrame)
        self.graphCanvas.draw()
        self.graphCanvas.get_tk_widget().pack(side=tk.BOTTOM,
                                              fill=tk.BOTH,
                                              expand=True)


# Define Main of Application
if __name__ == "__main__":
    # Where the Application is run from
    matplotlib.use("TkAgg")
    root = tk.Tk()
    app = Application(root)
    app.mainloop()
