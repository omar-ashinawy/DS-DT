import numpy as np
from sklearn.tree import DecisionTreeClassifier as sk_DT
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import pandas as pd
from tkinter import IntVar, Label, Entry, Tk, END, ALL, Toplevel, Button, Radiobutton, messagebox, filedialog, LabelFrame, StringVar
import os
#######################################################################################################
def entropy (labels):
    #Calculates entropy for two class table.
    n = len(labels)
    numFirstClass = np.sum((labels == 1))
    if (numFirstClass == 0 or numFirstClass == n):
        entropy = 0
    else:
        probFirstClass = numFirstClass / n
        probSecondClass = 1 - probFirstClass
        entropy = -(probFirstClass*np.log2(probFirstClass) + probSecondClass*np.log2(probSecondClass))
    return entropy

class TreeNode:
    def __init__(self, depth = 1, maxDepth = None):
        self.depth = depth
        self.maxDepth = maxDepth
        
    def train(self, X, labels):
        #first base case: checks if labels is one element or labels has one class (no possible splits) 
        if len(labels) == 1 or len(set(labels)) == 1:
            self.colmn = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = labels[0]
        else:
            numCols = range(X.shape[1])
            
            maxInfGain = 0
            bestCol = None
            bestColSplit = None
            for col in numCols:
                infGain, split = self.splitColoumn(X, labels, col)
                if infGain > maxInfGain:
                    maxInfGain = infGain
                    bestCol = col
                    bestColSplit = split
            
            if maxInfGain == 0: #no splits 
                self.colmn = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(labels.mean())
            else:
                self.colmn = bestCol
                self.split = bestColSplit
                
                if self.depth == self.maxDepth:
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.round(np.mean(labels[X[:,bestCol] < self.split])), #left node pred
                        np.round(np.mean(labels[X[:,bestCol] >= self.split])), #right node pred
                    ]
                else:
                    leftInd = (X[:, bestCol] < bestColSplit)
                    Xleft = X[leftInd]
                    labelsLeft = labels[leftInd]
                    self.left = TreeNode(self.depth + 1, self.maxDepth)
                    self.left.train(Xleft, labelsLeft) #recursion
                    
                    rightInd = (X[:, bestCol] >= bestColSplit)
                    Xright = X[rightInd]
                    labelsright = labels[rightInd]
                    self.right = TreeNode(self.depth + 1, self.maxDepth)
                    self.right.train(Xright, labelsright) #recursion

    def informationGain(self, colX, colY, split):
        yLeft = colY[colX < split]
        yRight = colY[colX >= split]
        yLen = len(colY)
        yLeftLen = len(yLeft)
        if (yLeftLen == 0 or yLeftLen == yLen):
            return 0
        yLeftProp = float(yLeftLen) / yLen
        yRightProp = 1 - yLeftProp
        return entropy(colY) - yLeftProp*entropy(yLeft) - yRightProp*entropy(yRight)
        
    def splitColoumn(self, X, labels, col):
        colToBeSplit_X = X[:, col]
        sortedInd = np.argsort(colToBeSplit_X)
        colToBeSplit_X = colToBeSplit_X[sortedInd]
        colToBeSplit_Y = labels[sortedInd]
        
        boundaries = np.nonzero(colToBeSplit_Y[:-1] != colToBeSplit_Y[1:])[0]
        
        best_split = None
        max_ig = 0
        for b in boundaries:
            split = (colToBeSplit_X[b] + colToBeSplit_X[b+1]) / 2
            ig = self.informationGain(colToBeSplit_X, colToBeSplit_Y, split)
            if ig > max_ig:
                max_ig = ig
                best_split = split
        
        return max_ig, best_split
        
    def predict_element(self, element):
        if self.colmn is not None and self.split is not None:
            if element[self.colmn] < self.split:
                if self.left:
                    p = self.left.predict_element(element)
                else:
                    p = self.prediction[0]
            else:
                if self.right:
                    p = self.right.predict_element(element)
                else:
                    p = self.prediction[1]
        else:
            p = self.prediction
        return p
    
    def predict(self, X):
        predictions = np.zeros(len(X)) #np.zeros_like(X)
        for i in range (len(X)):
            predictions[i] = self.predict_element(X[i])
        return predictions

class DecisionTreeClassifier:
    def __init__(self, maxDepth = None):
        self.maxDepth = maxDepth
    
    def fit(self, X, labels):
        #We called it fit to have the same name as sklearn DT model to be convenient to use.
        self.root = TreeNode(maxDepth = self.maxDepth)
        self.root.train(X, labels)
        
    def predict(self, X):
        return self.root.predict(X)
    
    def accuracy(self, X, labels, metric = "accuracy"):
        pred = self.predict(X)
        if metric == "accuracy":
            return np.mean(pred == labels)*100
        elif metric == "f1_score":
            return f1_score(labels, pred, average = "weighted")*100
        elif metric == "matthews":
            return matthews_corrcoef(labels, pred)*100
#######################################################################################################

def pretext(event):
    global max_depth_text
    if event == '<Enter>':
        if max_depth_entry.get() == "Specify maximum depth":
            max_depth_entry.delete(0, END)
    elif event == '<Leave>':
        max_depth_text = max_depth_entry.get()
        if max_depth_entry.get() == "":
            max_depth_entry.insert(0, "Specify maximum depth")

def open_train(choice = 0, max_depth_text = None, metric = "accuracy"):
    global y_train_enc
    global our_classifier
    global sk_classifier
    global flag 
    if not max_depth_text.isdigit() and max_depth_text != "None":
        messagebox.showerror("Error", "Max Depth Must Be an Integer!\nهو حضرتك بتعمل ايه ؟")
    else:
        if max_depth_text == "None":
            max_depth_int = None
        else:
            max_depth_int = int(max_depth_text)
        root.filename = filedialog.askopenfilename(title = "Select Training File", initialdir = "/", filetypes = (("CSV File", "*.csv"), ("All Files", "*.*")) )
        if root.filename == "":
            messagebox.showerror("Error", "No File was selected!\nهو حضرتك بتعمل ايه ؟")
        else:
            new_window = Toplevel()
            new_window.title("Please wait...")
            Label(new_window, text = "Please wait, Our Decision Tree is being trained!").pack()
            new_window.update()
            dataset1 = pd.read_csv(root.filename)
            X_train = dataset1.iloc[:, :-2].values
            y_train = dataset1.iloc[:, -1].values
            y_train_enc = LabelEncoder()
            y_train = y_train_enc.fit_transform(y_train)
            our_classifier = DecisionTreeClassifier(maxDepth = max_depth_int)
            our_classifier.fit(X_train, y_train)
            our_train_acc = our_classifier.accuracy(X_train, y_train, metric)
            flag = 1
            new_window.destroy()
            if choice == 0:
                messagebox.showinfo(title = "Training Accuracy", message = "Accuracy on our Classifier = " + str(our_train_acc))
            elif choice == 1:
                new_window_sk = Toplevel()
                new_window_sk.title("Please wait...")
                Label(new_window_sk, text = "Please wait, Sklearn Decision Tree is being trained!").pack()
                new_window_sk.update()
                sk_classifier = sk_DT(criterion = "entropy", max_depth = max_depth_int, random_state = 0)
                sk_classifier.fit(X_train, y_train)
                if metric == "accuracy":
                    sk_train_acc = np.mean(sk_classifier.predict(X_train) == y_train)*100
                elif metric == "f1_score":
                    sk_train_acc = f1_score(y_train, sk_classifier.predict(X_train), average = "weighted")*100
                elif metric == "matthews":
                    sk_train_acc = matthews_corrcoef(y_train, sk_classifier.predict(X_train))*100 
                new_window_sk.destroy()
                messagebox.showinfo(title = "Training Accuracy", message = ("Accuracy on our Classifier = " + str(our_train_acc) + "\nAccuracy on Sklearn Classifier = " + str(sk_train_acc)))

def open_validate(choice = 0, metric = "accuracy"):
    if flag == 0 and not max_depth_text.isdigit():
        messagebox.showerror(title = "Error", message = "You must train the classifier first!\n هو حضرتك بتعمل ايه ؟")
    else:
        filename = filedialog.askopenfilename(title = "Select Validation File", initialdir = "/", filetypes = (("CSV File", "*.csv"), ("All Files", "*.*")) )
        if filename == "":
            messagebox.showerror(title = "Error", message = "No file was selected!\n هو حضرتك بتعمل ايه ؟")
        else:
            new_window = Toplevel()
            new_window.title("Please wait...")
            Label(new_window, text = "Please wait, Our Decision Tree is being validated!").pack()
            new_window.update()
            dataset2 = pd.read_csv(filename)
            X_test = dataset2.iloc[:, :-2].values
            y_test = dataset2.iloc[:, -1].values
            y_test_enc = LabelEncoder()
            y_test = y_train_enc.fit_transform(y_test)
            our_test_acc = our_classifier.accuracy(X_test, y_test, metric)
            new_window.destroy()
            if choice == 0:
                messagebox.showinfo(title = "Validation Accuracy", message = "Accuracy on our Classifier = " + str(our_test_acc))
            elif choice == 1:
                new_window_sk = Toplevel()
                new_window_sk.title("Please wait...")
                Label(new_window_sk, text = "Please wait, Sklearn Decision Tree is being validated!").pack()
                new_window_sk.update()
                if metric == "accuracy":
                    sk_test_acc = np.mean(sk_classifier.predict(X_test) == y_test)*100
                elif metric == "f1_score":
                    sk_test_acc = f1_score(y_test, sk_classifier.predict(X_test), average = "weighted")*100
                elif metric == "matthews":
                    sk_test_acc = matthews_corrcoef(y_test, sk_classifier.predict(X_test))*100 
                new_window_sk.destroy()
                messagebox.showinfo(title = "Validation Accuracy", message = ("Accuracy on our Classifier = " + str(our_test_acc) + "\nAccuracy on Sklearn Classifier = " + str(sk_test_acc)))

def save_test():
    if flag == 0 and not max_depth_text.isdigit():
        messagebox.showerror(title = "Error", message = "You must train the classifier first!\n هو حضرتك بتعمل ايه ؟")
    else:
        testfile = filedialog.askopenfilename(title = "Select Test File", initialdir = "/", filetypes = (("CSV File", "*.csv"), ("All Files", "*.*")) )
        dataset2 = pd.read_csv(testfile)
        X_test = dataset2.iloc[:, :-2].values
        y_test = our_classifier.predict(X_test)
        messagebox.showinfo(title = "Info", message = "Choose directory to save test results in")
        dir_name = filedialog.askdirectory(title = "Choose Directory", initialdir = "/")
        file_name = os.path.join(dir_name, "test_results.txt")
        file = open(file_name, mode = "w")
        for i in range(len(y_test)):
            file.write(str(y_test[i]) + "\n")
        file.close()
        messagebox.showinfo(title = "Success!", message = "Test results saved in: " + file_name)

#######################################################################################################
if __name__ == "__main__":
    root = Tk()
    root.title("Decision Tree Classifier")
    root.geometry("400x400")
    flag = 0

    max_depth_text = ""

    radio_choice = IntVar()
    radio_choice.set(0)

    radio_acc = StringVar()
    radio_acc.set("accuracy")

    max_depth_entry = Entry(root, width= 170, justify = "c")
    max_depth_entry.insert(0, "Specify maximum depth")
    max_depth_entry.bind(sequence = '<Enter>', func = lambda event: pretext('<Enter>'))
    max_depth_entry.bind(sequence = '<Leave>', func = lambda event: pretext('<Leave>'))
    max_depth_entry.pack()

    Radiobutton(root, text = "Compare with Sklearn Classifier", value = 1, variable = radio_choice).pack()
    Radiobutton(root, text = "Only Train On Our Classifier", value = 0, variable = radio_choice).pack()

    acc_frame = LabelFrame(root, text = "Accuracy Metric")
    Radiobutton(acc_frame, text = "Usual Accuracy", value = "accuracy", variable = radio_acc).pack(side = "left")
    Radiobutton(acc_frame, text = "F1 Score", value = "f1_score", variable = radio_acc).pack(side = "left")
    Radiobutton(acc_frame, text = "Matthews Correlation Coeffcient", value = "matthews", variable = radio_acc).pack(side = "left")
    acc_frame.pack()

    open_btn = Button(root, text = "Choose a training set", command = lambda: open_train(radio_choice.get(), max_depth_text, radio_acc.get()), width = 170).pack()
    validate_btn = Button(root, text = "Choose a validation set", command = lambda: open_validate(radio_choice.get(), radio_acc.get()), width = 170).pack()
    test_btn = Button(root, text = "Save test results", command = save_test, width = 170).pack()

    root.mainloop()

