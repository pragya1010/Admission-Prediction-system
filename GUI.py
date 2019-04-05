from tkinter import *
import project
import pandas as pd


ad = project.Admission_Predictor()
# print(ad.data.columns)
l=[]
def fetch(entries):
      for entry in entries:
         field = entry[0]
         text  = entry[1].get()
         # print('%s: "%s"' % (field, text))
         l.append(float(text))
         # print(l)


      df= pd.DataFrame(columns=['A','B','C','D','E','F','G'])
      df = df.append(pd.DataFrame([l],columns=df.columns))
      # print(df)
      pred = ad.predict(df)
      decision = "High" if pred[0] > 0.80 else "Low"
      label.config(text="Predicted Acceptance rate :{}% \n Chances are: {}".format(round(pred[0],2)*100,decision))


def makeform(root, fields):
   entries = []
   for field in fields:
      row = Frame(root)
      lab = Label(row, width=15, text=field, anchor='w')
      ent = Entry(row)
      row.pack(side=TOP, fill=X, padx=5, pady=5)
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES, fill=X)
      entries.append((field, ent))
   return entries


fields = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
root = Tk()
ents = makeform(root, fields)
root.title("My Application")
root.bind('<Return>', (lambda event, e=ents: fetch(e)))
b1 = Button(root, text='Predict',
          command=(lambda e=ents: fetch(e)))
b1.pack(side=LEFT, padx=5, pady=5)
b2 = Button(root, text='Quit', command=root.quit)
b2.pack(side=LEFT, padx=5, pady=5)

label = Label(root, fg="green")
label.pack(side=LEFT, padx=5, pady=5)

root.mainloop()
