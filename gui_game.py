from tkinter import *

# Gui
window = Tk()
window.geometry("250x250")
window.title('Jouer')

def onClick(args):
    if args == 1:
        window.destroy()
        import gui_register
    if args == 2:
        window.destroy()
        import gui_connect

# register btn
Register = Button(window, text="Registration", command=lambda:onClick(1)).place(x=75, y=100)
# connection btn
Connect = Button(window, text="Connection", command=lambda:onClick(2)).place(x=75, y=125)

window.mainloop()
