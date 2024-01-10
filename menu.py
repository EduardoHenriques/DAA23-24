import threading
import tkinter as tk

def run_menu(modelo1_LinearReg, modelo1_2_PolyReg, modelo2_DecisionTree, modelo3_RandomForest, modelo4_MLP, modelo5_MLP_Classify, data):
    def run_model1():
        modelo1_LinearReg(data)

    def run_model2():
        modelo1_2_PolyReg(data)

    def run_model3():
        modelo2_DecisionTree(data)
        
    def run_model4():
        modelo3_RandomForest(data)

    def run_model5():
        modelo4_MLP(data)
    
    def run_model6():
        modelo5_MLP_Classify(data)
        
    def run_exit():
        exit()
    
    def run_gui():
        root = tk.Tk()
        root.title("DAA 23-24  | Menu ")
        root.geometry("400x400")  # Set window size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        # calculate position x and y coordinates
        x = (screen_width/2) - (400/2)
        y = (screen_height/2) - (400/2)
        root.geometry('%dx%d+%d+%d' % (400, 400, x, y))  # Center window
        
        button1 = tk.Button(root, text="Linear Regression", command=run_model1)
        button1.pack(fill=tk.X, padx=50, pady=10)

        button2 = tk.Button(root, text="Polynomial Regression", command=run_model2)
        button2.pack(fill=tk.X, padx=50, pady=10)

        button3 = tk.Button(root, text="Decision Tree", command=run_model3)
        button3.pack(fill=tk.X, padx=50, pady=10)

        button4 = tk.Button(root, text="Random Forest", command=run_model4)
        button4.pack(fill=tk.X, padx=50, pady=10)

        button5 = tk.Button(root, text="MLP", command=run_model5)
        button5.pack(fill=tk.X, padx=50, pady=10)

        button6 = tk.Button(root, text="MLP Classify", command=run_model6)
        button6.pack(fill=tk.X, padx=50, pady=10)
        
        button6 = tk.Button(root, text="Exit", command=run_exit)
        button6.pack(fill=tk.X, padx=50, pady=10)

        root.mainloop()
    threading.Thread(target=run_gui).start()