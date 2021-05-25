# coding: utf-8
from tkinter import *
import mysql.connector

mydb = mysql.connector.connect(
    host="127.0.0.1",
    port=8889,
    user="root",
    password="root",
    database="Software"
)

mycursor = mydb.cursor()

# Creating object 'window' of Tk()
window = Tk()
window.geometry("500x500")
window.title('Registration')

def onClick(args):
    if args == 1:
        window.destroy()
        import gui_game
    if args == 2:
        sql = "INSERT INTO Users (FirstName, LastName, UserName, Email, PassWord) VALUES (%s, %s, %s, %s, %s)"
        val = (FirstName, LastName, UserName, Email, PassWord)
        mycursor.execute(sql, val)
        mydb.commit()

        #window.destroy()
        #import gui_Play

# this creates 'Label' widget for Registration Form and uses place() method.
label_0 = Label(window, text="Registration form", width=20, font=("bold", 20))
label_0.place(x=125, y=60)

label_1 = Label(window, text="First Name", width=20, font=("bold", 10))
label_1.place(x=80, y=130)

label_2 = Label(window, text="Last Name", width=20, font=("bold", 10))
label_2.place(x=80, y=170)

label_3 = Label(window, text="User Name", width=20, font=("bold", 10))
label_3.place(x=80, y=200)

label_4 = Label(window, text="Email Address", width=20, font=("bold", 10))
label_4.place(x=80, y=240)

label_5 = Label(window, text="Password", width=20, font=("bold", 10))
label_5.place(x=80, y=280)

entry_1 = Entry(window)
entry_1 .place(x=240, y=130)
FirstName = entry_1.get()

entry_2 = Entry(window)
entry_2.place(x=240, y=170)
LastName = entry_2.get()

entry_3 = Entry(window)
entry_3.place(x=240, y=200)
UserName = entry_3.get()

entry_4 = Entry(window)
entry_4.place(x=240, y=240)
Email = entry_4.get()

entry_5 = Entry(window)
entry_5.place(x=240, y=280)
PassWord = entry_5.get()

# this creates button for submitting the details provides by the user
Button(window, text='Submit', width=20, bg="white", fg='black', command=lambda:onClick(2)).place(x=180, y=380)
Button(window, text='Go back', width=20, bg='white', fg='red', command=lambda:onClick(1)).place(x=180, y=410)

# this will run the mainloop.
window.mainloop()
