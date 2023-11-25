from tkinter import * 
from tkinter import messagebox
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
win=Tk()
win.state("zoomed")
win.resizable(width=False,height=False)
win.configure(bg="orange")
win.title("My project")

lbl_title=Label(win,text="Face Detection",font=('',55,'bold','underline'),bg='orange')
lbl_title.pack()
eyesModel=cv2.CascadeClassifier("haarcascade_eye.xml")

#----------------------------------------Image---------------------------------------
imageFrame=None
def startface(frame,cv2image,lmain):
    clf=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    faces=clf.detectMultiScale(gray,1.2,5)
    eyes=eyesModel.detectMultiScale(gray)
    
    for x,y,w,h in faces:
        cv2.rectangle(cv2image,(x,y),(x+w,y+h),(255,0,0),2)
        for ex,ey,ew,eh in eyes:
            pass
            #cv2.rectangle(cv2image,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
def browse():
    global imageFrame
    if(imageFrame!=None):
        imageFrame.destroy()
    file_path=filedialog.askopenfilename()
    frame=cv2.imread(file_path)
    cv2image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    imageFrame= Frame(width=700, height=500,bd=6,bg='black')
    lmain =Label(imageFrame)
    lmain.grid(row=0, column=0)
    imageFrame.place(relx=.37,rely=.3)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    btn_Detection=Button(command=lambda:startface(frame,cv2image,lmain),text='Detect face',font=('',20,'bold'),bd=10,width=12).place(relx=.4,rely=.9)
def image_screen():
    frm=Frame(win,bg='sky blue')
    frm.place(relx=0,rely=.15,relwidth=1,relheight=1)
 
    btn_Browse=Button(frm,command=lambda:browse(),text="Browse",font=('',20,'bold'),bd=10,width=8)
    btn_Browse.place(relx=.38,rely=.01)

    btn_back=Button(frm,command=lambda:welcome_screen(),text="back",font=('',20,'bold'),bd=10)
    btn_back.place(relx=.9,rely=0)
#----------------------------------------------xxxxxxxxxx-----------------------------------------------------------    


#-------------------------------------------Video_Screen--------------------------------------------------------------
iFrame=None
def browse_video():
    global iFrame
    if(iFrame!=None):
        iFrame.destroy()
    file_path=filedialog.askopenfilename()
    vdo=cv2.VideoCapture(file_path)
    while(True):
        flag,img=vdo.read()
        if(flag==True):
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            eyes=eyesModel.detectMultiScale(gray)
            faces=clf.detectMultiScale(gray,1.2,6);cv2.putText(img,"Press 'q' to quit video",(20,30),5,2,(255,0,0),2)
            for x,y,w,h in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                for ex,ey,ew,eh in eyes:
                    pass
                    #cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
            cv2.imshow('vdo',img)
            k=cv2.waitKey(10)
            if(k==ord('q')):
                break  
        else:
            break
    vdo.release()
    cv2.destroyAllWindows()
    
    
def video_screen(): 
    frm=Frame(win,bg='sky blue')
    frm.place(relx=0,rely=.15,relwidth=1,relheight=1)
  
    btn_Browse=Button(frm,command=lambda:browse_video(),text="Browse",font=('',20,'bold'),bd=10,width=8)
    btn_Browse.place(relx=.37,rely=.05)
    btn_back=Button(frm,command=lambda:welcome_screen(),text="back",font=('',20,'bold'),bd=10)
    btn_back.place(relx=.9,rely=0)
    

    
    
    
    

#----------------------------------------xxxxxxxxxxxxxxxxxxxxxxxxxxxx-----------------------------------------------   
    
   




    
    
    
#--------------------------------------------WebCam-----------------------------------------------------------------
flag=False  
clf=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def stop():
    cap.release()
    lmain.destroy()
    imageFrame.destroy()

def startfac():
    global flag
    flag=True

def stopface():
    global flag
    flag=False

def show_frame():
    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    eyes=eyesModel.detectMultiScale(gray)
    faces=clf.detectMultiScale(gray,1.3,7)
    for x,y,w,h in faces:
        if(flag==True):
            cv2.rectangle(cv2image,(x,y),(x+w,y+h),(255,0,0),2)
            for ex,ey,ew,eh in eyes:
                pass
                    #cv2.rectangle(cv2image,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame) #calls show_frame after 10 mills

def start():
    global lmain,cap,lmain,imageFrame
    cap=cv2.VideoCapture(0)
    imageFrame=Frame(width=1500,height=2000,bd=1,bg='green')
    lmain=Label(imageFrame)
    lmain.grid(row=5, column=10)
    imageFrame.place(relx=.3,rely=.3)
    btn_DetectFace=Button(imageFrame,command=lambda:startfac(),text='detect face',font=('',20,'bold'),bd=10,width=12).place(relx=.1,rely=.8)
    btn_Stop=Button(imageFrame,command=lambda:stopface(),text='stop detecting',font=('',20,'bold'),bd=10,width=12).place(relx=.5,rely=.8)
    
    show_frame()
    
def webcam_screen():
    frm=Frame(win,bg='sky blue')
    frm.place(relx=0,rely=.15,relwidth=1,relheight=1)
    btn_StartCam=Button(command=lambda:start(),text='start camera ',font=('',20,'bold'),bd=10).place(relx=.01,rely=.2)
    btn_StopCam=Button(command=lambda:stop(),text='stop camera',font=('',20,'bold'),bd=10).place(relx=.01,rely=.4)
    btn_back=Button(frm,command=lambda:welcome_screen(),text="back",font=('',20,'bold'),bd=10)
    btn_back.place(relx=.9,rely=0)
    
#-------------------------------------------------xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx------------------------------------------







    


def logout():
    option=messagebox.askyesno('Confirmation','Do you want to logout?')
    if(option==True):
        home_screen()
    else:
        pass


def home_screen():
    frm=Frame(win,bg='sky blue')
    frm.place(relx=0,rely=.15,relwidth=1,relheight=1)
    
    lbl_user=Label(frm,text="Username",font=('',20,'bold'),bg='sky blue')
    lbl_user.place(relx=.28,rely=.3)

    entry_user=Entry(frm,font=('',20,'bold'),bd=10)
    entry_user.place(relx=.42,rely=.3)
    entry_user.focus()

    lbl_pass=Label(frm,text="Password",font=('',20,'bold'),bg='sky blue')
    lbl_pass.place(relx=.28,rely=.4)

    entry_pass=Entry(frm,font=('',20,'bold'),bd=10,show="*")
    entry_pass.place(relx=.42,rely=.4)

    btn_login=Button(frm,command=lambda:welcome_screen(entry_user,entry_pass),text="login",font=('',20,'bold'),bd=10,width=10)
    btn_login.place(relx=.45,rely=.5)
        
    
def welcome_screen(entry_user=None,entry_pass=None):
    if(entry_user!=None and entry_pass!=None):
        user=entry_user.get()
        pwd=entry_pass.get()
    else:
        user="admin"
        pwd="admin"
    if(len(user)==0 or len(pwd)==0):
        messagebox.showwarning("validation","Please fill both fields")
        return
    else:
        if(user=="admin" or pwd=="admin"):
            frm=Frame(win,bg='skyblue')
            frm.place(relx=0,rely=.15,relwidth=1,relheight=1)

            btn_image=Button(frm,command=lambda:image_screen(),text="Use Image",font=('',20,'bold'),bd=10,width=25)
            btn_image.place(relx=.3,rely=.2)

            btn_video=Button(frm,command=lambda:video_screen(),text="Use Video",font=('',20,'bold'),bd=10,width=25)
            btn_video.place(relx=.3,rely=.4)
            
            btn_webcam=Button(frm,command=lambda:webcam_screen(),text="Use Webcam",font=('',20,'bold'),bd=10,width=25)
            btn_webcam.place(relx=.3,rely=.6)

            btn_logout=Button(frm,command=lambda:logout(),text="logout",font=('',20,'bold'),bd=10)
            btn_logout.place(relx=.88,rely=0)
        else:
            messagebox.showerror("Fail","Invalid Username/Password")    
    
    
home_screen()
win.mainloop()
