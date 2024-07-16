import tkinter as tk
import psycopg2
from tkinter import messagebox 

def insertdata(name,phone):
    # connection = psycopg2
    connection= psycopg2.connect(
        host = "localhost",
        user = "postgres",
        password = "9509",
        dbname =  "postgres",
        port = 5432
    ) 
    print(f"Name : {name},Phone : {phone}")
    cursor = connection.cursor()
    cursor.execute("insert into details(name,phone) values (%s,%s)",(name,phone))
    connection.commit()
    messagebox.showinfo("success")
    cursor.close()
    connection.close()
    


def myfun():
    name = name_entry.get()
    phone = phone_entry.get()
    print(f"Name :{name},Phone :{phone}")
    insertdata(name,phone)
    messagebox.showinfo("submitted","details submitted")
    



root = tk.Tk()
root.title("use details form")

tk.Label(root,text ="name:").grid(row=0,column=0,padx=0,pady=10)
name_entry = tk.Entry(root)
name_entry.grid(row=0,column=1,padx=10,pady=10)

tk.Label(root,text ="phone:").grid(row=1,column=0,padx=0,pady=10)
phone_entry = tk.Entry(root)
phone_entry.grid(row=1,column=1,padx=10,pady=10)

submit_button = tk.Button(root,text="submit",command=myfun)
submit_button.grid(row=2,columnspan=2,pady=10)

root.mainloop()