import tkinter

# init tk
root = tkinter.Tk()

# create canvas
myCanvas = tkinter.Canvas(root, bg="white", height=300, width=300)

# draw arcs
coord = 10, 10, 300, 300
arc = myCanvas.create_arc(coord, start=0, extent=150, fill="red")
arv2 = myCanvas.create_arc(coord, start=150, extent=215, fill="green")

# add to window and show
myCanvas.pack()
root.mainloop()
