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
        sql = "SELECT Email, PassWord FROM Users WHERE LOWER(UserName) LIKE %s"
        val = (UserName)
        mycursor.execute(sql, val)
        Result = mydb.fetchone()

        # window.destroy()
        # import gui_Play


label_0 = Label(window, text="Connection form", width=20, font=("bold", 20))
label_0.place(x=125, y=60)

label_1 = Label(window, text="User Name", width=20, font=("bold", 10))
label_1.place(x=80, y=150)

label_2 = Label(window, text="Email Address", width=20, font=("bold", 10))
label_2.place(x=80, y=190)

label_3 = Label(window, text="Password", width=20, font=("bold", 10))
label_3.place(x=80, y=220)

entry_1 = Entry(window)
entry_1.place(x=240, y=150)
UserName = entry_1.get()

entry_2 = Entry(window)
entry_2.place(x=240, y=190)
email = entry_2.get()

entry_3 = Entry(window)
entry_3.place(x=240, y=220)
passWord = entry_3.get()

# this creates button for submitting the details provides by the user
Button(window, text='Submit', width=20, bg='white', fg='black').place(x=180, y=300)
Button(window, text='Go back', width=20, bg='white', fg='red', command=lambda:onClick(1)).place(x=180, y=330)
# this will run the mainloop.
window.mainloop()
