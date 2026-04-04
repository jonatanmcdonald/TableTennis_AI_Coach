import tkinter as tk
from tkinter import font
import threading
import RealTimeFeedback2D

def start_feedback(feedback_ui_callback=None):
    if not RealTimeFeedback2D.running:
        threading.Thread(target=RealTimeFeedback2D.start_feedback).start()


def stop_feedback():
    RealTimeFeedback2D.stop_feedback()


#create main window
root = tk.Tk()
root.title("Your Table Tennis AI Coach!")
root.geometry("500x500")
root.configure(bg="#1a1a2e") #dark blue

#custom fonts
title_font = font.Font(family="Helvetica", size=20, weight="bold")
btn_font = font.Font(family="Helvetica", size=14, weight="bold")
feedback_font = font.Font(family="Helvetica", size=16)

#title label
title = tk.Label(root, text="TT AI Coach", font=title_font, bg="#1a1a2e", fg="#e94560") #pink
title.pack(pady=20)

#feedback label
feedback_label = tk.Label(root, text="Feedback will appear here", font=feedback_font, bg="#162447", fg="#f0f0f0")
feedback_label.pack(pady=10, ipadx=10, ipady=10, fill="x")

def update_feedback(feedback):
    feedback_label.config(text=feedback)


#Buttons frame
btn_frame = tk.Frame(root, bg="#1a1a2e")
btn_frame.pack(pady=20)


#start button
start_btn = tk.Button(
    btn_frame, text="Start Coaching", font=btn_font,
    bg="#e94560", fg="white", activebackground="#ff2e63",
    activeforeground="white", width=15, height=2,
    command=lambda: threading.Thread(target=RealTimeFeedback2D.start_feedback, args=(update_feedback,)).start() 
                    )
start_btn.grid(row=0, column=0, padx=0)

#stop button
stop_btn = tk.Button(btn_frame, text="Stop Coaching", font=btn_font, bg="#0f3460", fg="white", activebackground="#162447",
                     activeforeground="white", width=15, height=2, command=stop_feedback)
stop_btn.grid(row=1, column=0, padx=0, pady=10)

#footer
footer_label = tk.Label(root,text="Use ESC to exit the live camera", font=("Helvetica", 10), bg="#1a1a2e", fg="#999")
footer_label.pack(pady=20)


#Run UI
root.mainloop()