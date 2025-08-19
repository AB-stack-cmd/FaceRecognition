import sqlite3 as db
from datetime import datetime

# to connect with database else make a database if doesnot exist
data = db.connect("my_database.db")

cursor = data.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTs users(
               NAME VARCHAR(50),
               TIME TEXT
               )
               """)
def make_attendence(name,id):
    # id =  input("Enter id")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute( "INSERT INTO users values (?,?)", (name,now))

make_attendence("rohit",12)