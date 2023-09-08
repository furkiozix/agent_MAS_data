# -*- coding: utf-8 -*-
from pade.misc.utility import display_message, start_loop
from pade.core.agent import Agent
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import FipaContractNetProtocol
from sys import argv
from random import uniform
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from pandas import read_csv
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import random
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import  precision_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss
import pickle
from sklearn.utils import shuffle

from sklearn.metrics import f1_score


#class for load dataset to the memmory
class datasetc:
    #Varriable for trainning
    X = np.array([0])
    Y = np.array([0])
    X_test = np.array([0])
    X_train = np.array([0])
    Y_train = np.array([0])
    Y_test = np.array([0])
    X_val=np.array([0])
    Y_val=np.array([0])

    def __init__(self):
        print("nothingtodo")

    @staticmethod
    def datagiv():
        #Load dataset
        filename = 'csvfile/pe_section_headers.csv'

        dataframe = read_csv(filename)

        # Convert dataframe to numpy array


        le = LabelEncoder()
        for col in dataframe.columns:
            if dataframe[col].dtypes == 'object':  ####### when column's data type is equal to object
                dataframe[col] = le.fit_transform(dataframe[col])  ###### fit_transform is used for conversion
        dataframe=dataframe.drop(['hash'],axis=1)
        dataframe = dataframe.drop(['virtual_address'], axis=1)
        dataframe=shuffle(dataframe)
        train, validate, test = np.split(dataframe.sample(frac=1), [int(.7 * len(dataframe)), int(.85 * len(dataframe))])
        train=train.values
        test=test.values
        validate=validate.values
        datasetc.X_train=train[:,0:3]
        datasetc.X_test=test[:,0:3]
        datasetc.Y_train=train[:,3]
        datasetc.Y_test =test[:,3]
        datasetc.X_val = validate[:,0:3]
        datasetc.Y_val = validate[:,3]


#Class of Model
class ClassifierML:

    def __init__(self, MLid):
        self.MLid = MLid

    def classification(self, model):
        return model.predict_proba(datasetc.X_test)
    def Class_confussionmatrix(self, model):
        yhat=model.predict(datasetc.X_test)
        return confusion_matrix(datasetc.Y_test, yhat)

    def evaluation_for_weights(self, model, strategy):
        # model Evaluation
        # Strategy choose
       
       
        Y_scores = model.predict_proba(datasetc.X_test)
        Y_pred=model.predict(datasetc.X_test)
        f1 = f1_score(datasetc.Y_test, Y_pred, average='weighted')
        print("F1 Score:", f1)


        if strategy == 1:
            fpr, tpr, threshold = roc_curve(datasetc.Y_test, Y_scores[:, 1])
            result = auc(fpr, tpr) #AUC
        elif strategy == 2:
            result = model.score(datasetc.X_val,datasetc.Y_val) #Accuracy
        elif  strategy == 3:
            result = precision_score(datasetc.Y_val,Y_pred, average='weighted') #Precision score
        elif strategy == 4:
            cm=confusion_matrix(datasetc.Y_val,model.predict(datasetc.X_test))
            TP = cm[1][1]
            TN = cm[0][0]
            FP = cm[0][1]
            FN = cm[1][0]
            result=(FP+FN)/(TP+TN+FP+FN) #Classification error rate

        # scores = cross_val_score(model, datasetc.X, datasetc.Y, cv=5)
        # t_pred = cross_val_predict(model, datasetc.X, datasetc.Y, cv=3, method='predict_proba')
        return result

    #KNN
    def knnpima(self):
        model = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
        model.fit(datasetc.X_train, datasetc.Y_train)
        return model
    # Gaussian
    def nbayiris(self):
        gnb = GaussianNB()
        gnb = gnb.fit(datasetc.X_train, datasetc.Y_train)
        return gnb
    #decision tree
    def decitree(self):
        clf2 = tree.DecisionTreeClassifier()
        clf2 = clf2.fit(datasetc.X_train, datasetc.Y_train)
        return clf2
    #random forest
    def ranfores(self):
        Random_Forest_model = RandomForestClassifier(n_estimators=5)
        model = Random_Forest_model.fit(datasetc.X_train, datasetc.Y_train)
        return model
    #boost algorithm
    def boosmodel(self):
        seed = 3
        num_trees = 10
        model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
        model = model.fit(datasetc.X_train, datasetc.Y_train)
        return model
    #bagging
    def bagging(self):
        seed = 3
        cart = DecisionTreeClassifier()
        num_trees = 50
        model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed).fit(datasetc.X_train, datasetc.Y_train)
        return model
    #LDA
    def LDA(self):
        classifier = LinearDiscriminantAnalysis(n_components=1)
        classifier = classifier.fit(datasetc.X_train, datasetc.Y_train)
        return classifier



#Master agent Behavior
class CompContNet1(FipaContractNetProtocol):
    '''CompContNet1

       Initial FIPA-ContractNet Behaviour that sends CFP messages
       to other feeder agents asking for restoration proposals.
       This behaviour also analyzes the proposals and selects the
       one it judges to be the best.'''

    def __init__(self, agent, message):
        super(CompContNet1, self).__init__(
            agent=agent, message=message, is_initiator=True)
        self.cfp = message
    #This function will start after all agent sent all result to master agent
    def handle_all_proposes(self, proposes):
        """
        After all slave agent sent all data to Master agent , Master will calculate their weight
        and draw their ROC curve then Master will make ROC for Ensamble one
        """
        super(CompContNet1, self).handle_all_proposes(proposes)
        best_proposer = None #for the agent who choose
        best_auc=0.0
        other_proposers = list() #for other who didn't choose
        display_message(self.agent.aid.name, 'Analyzing.......')
        fpr=dict()
        tpr=dict()
        roc_auc=dict()
        eval_weight=dict()
        Collect_result=list()
        result_with_weight=list()
        
        
        i = 1
        labels = ['KNN', 'Naivebayes', 'Bagging', 'Boosting','Decision tree', 'RandomForest', 'LDA','Ensemble']
        s = np.array([], dtype=float)#summation of all prediction probabillity
        

        for message in proposes:
            content = message.content
            content = pickle.loads(content)
            
            f1_scores = []
            agent_names = []

            results = np.array(np.array(content.get('prob')))#get the result
            display_message(self.agent.aid.name,'Analyzing  {i}'.format(i=getattr(message.sender,'name')))
            # display_message(self.agent.aid.name,'Accuracy Offered: {pot}'.format(pot=power))
            #TODO or to correct
            #message nb 1 ( i == 1 ) ??
            # handle the message with weight {clasifier, weight}
            # store somewhere this weight
            # message nb 2
            # handle results of classification of test data
            curr_weight = float(content.get('weight'))
            eval_weight.update([(message.sender,curr_weight)]) #Collect weight and label
            Collect_result.append(results)
            """
            if i < 2:
                 s = np.vstack([s, results]) if s else results
                 #result_with_weight=results*curr_weight
            else:
                # s = np.vstack([s, results])
                 s=s+results
                 #result_with_weight=result_with_weight+(results*curr_weight)"""
            # use  probabilities for the positive outcome only
            fpr[i],tpr[i],_=roc_curve(datasetc.Y_test,results[:,1],pos_label=1) #Multiply all probability
            roc_auc[i]=auc(fpr[i],tpr[i])
            
            
            
            yhat = np.argmax(results, axis=1)
            f1 = f1_score(datasetc.Y_test, yhat, average='weighted')
            
            f1_scores.append(f1)
            agent_names.append(getattr(message.sender,'name'))
            
            
            # Individual model plot
            plt.plot(fpr[i], tpr[i], label='%s (AUC = %0.2f, F1 = %0.2f)' % (getattr(message.sender, 'name'), roc_auc[i], f1))
            

            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            plt.grid()
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

            other_proposers.append(message.sender)
            i += 1
        if other_proposers != []:
            weight_sum = sum(eval_weight.values()) #sum it
            eval_weight={k:v/weight_sum for k,v in eval_weight.items()} #calculate weight for each classifier: eval_weight/weight_sum
            #make weight to be np array for calculate AUC Probability
            eval_weight_np = np.array([eval_weight.get(a) for a in eval_weight])
            Collect_result=np.array(Collect_result)
            #Calculate AUC Probability
            k=0
            for k in range (i-1):
                  Collect_result[k]=Collect_result[k]*eval_weight_np[k]
            Auc_all=sum(Collect_result)

            fpr2, tpr2, _ = roc_curve(datasetc.Y_test,Auc_all[:,1],pos_label=1) # from all model
            roc_auc2 = auc(fpr2, tpr2)# from all model
            
            
            yhat_ensemble = np.argmax(Auc_all, axis=1)
            f1_ensemble = f1_score(datasetc.Y_test, yhat_ensemble, average='weighted')



            # Ensemble plot
            plt.plot(fpr2, tpr2, label='%s (AUC = %0.2f, F1 = %0.2f)' % ('Ensemble', roc_auc2, f1_ensemble), linestyle='-.', color='red')
            
            
            
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            plt.grid()
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')

            yhat = np.argmax(Auc_all, axis=1)
            # make confusion metrix
            cm = confusion_matrix(datasetc.Y_test, yhat)
            TP = cm[1][1]
            TN = cm[0][0]
            FP = cm[0][1]
            FN = cm[1][0]
            # Calculate Classification error
            loss =  (FP + FN) / (TP + TN + FP + FN)
            display_message(self.agent.aid.name,
                            'Confusion matrix is : {pot} '.format(
                                pot=cm))
            display_message(self.agent.aid.name,
                            'Classification error rate is : {pot} '.format(
                                pot=loss))

        plt.show()
        #for confirm about Agent that was choose for this data set
        if other_proposers!=[]:
            display_message(self.agent.aid.name,
                            'Sending ACCEPT_PROPOSAL answers...')
            answer = ACLMessage(ACLMessage.ACCEPT_PROPOSAL)
            answer.set_protocol(ACLMessage.FIPA_CONTRACT_NET_PROTOCOL)
            for k, v in eval_weight.items():
                answer.add_receiver(k)
            answer.set_content('Thanks for work hard')
            self.agent.send(answer)



    def handle_inform(self, message):
        """
        """
        super(CompContNet1, self).handle_inform(message)

        display_message(self.agent.aid.name, 'INFORM message received')

    def handle_refuse(self, message):
        """
        """
        super(CompContNet1, self).handle_refuse(message)

        display_message(self.agent.aid.name, 'REFUSE message received')

    def handle_propose(self, message):
        """
        """
        super(CompContNet1, self).handle_propose(message)

        display_message(self.agent.aid.name, 'PROPOSE message received')

#Slave agent behavior
class CompContNet2(FipaContractNetProtocol):
    '''CompContNet2

       FIPA-ContractNet Participant Behaviour that runs when an agent
       receives a CFP message. A proposal is sent and if it is selected,
       the restrictions are analized to enable the restoration.'''

    def __init__(self, agent, mlid,stategy):
        super(CompContNet2, self).__init__(agent=agent,
                                           message=None,
                                           is_initiator=False)
        self.stategy=stategy
        self.mlid = mlid #all agent will have individual id for assign their model


    def handle_cfp(self, message):
        """
        when start all the slave agent will get CFP message for initial them self
        after recieve message they will make a classification model and sent evaluation value
        to Master agent
        """
        super(CompContNet2, self).handle_cfp(message)
        self.message = message

        display_message(self.agent.aid.name, 'CFP message received')

        answer = self.message.create_reply()
        answer.set_performative(ACLMessage.PROPOSE)
        # Do classification
        self.modelmaker(self.mlid)
        # Prepare evaluation data
        All_data={'weight':self.calculateweight(),'prob':self.classification(),'cm':self.get_confm()}
        All_data_encode = pickle.dumps(All_data)
        answer.set_content(All_data_encode)
        # send a result
        self.agent.send(answer)

    def modelmaker(self, mlid):
        #assign type of model
        modeling = ClassifierML(1)
        if mlid == 0:
            display_message(self.agent.aid.name, 'This is K-nn model')
            self.b = modeling.knnpima()
        elif mlid == 1:
            display_message(self.agent.aid.name, 'This is Naive-Bayes model')
            self.b = modeling.nbayiris()
        elif mlid == 2:
            display_message(self.agent.aid.name, 'This is Bagging algorithm model')
            self.b = modeling.bagging()
        elif mlid == 3:
            display_message(self.agent.aid.name, 'This is Boosting algorithm model')
            self.b = modeling.boosmodel()
        elif mlid == 4:
            display_message(self.agent.aid.name, 'This is Decision tree model')
            self.b = modeling.decitree()
        elif mlid == 5:
            display_message(self.agent.aid.name, 'This is Random forest model')
            self.b = modeling.ranfores()
        else:
            display_message(self.agent.aid.name, 'This is  LDA model')
            self.b = modeling.LDA()


    def calculateweight(self):
        modeling = ClassifierML(1)
        self.weight = modeling.evaluation_for_weights(self.b,self.stategy)
        return self.weight

    def classification(self):
        modeling = ClassifierML(1)
        return modeling.classification(self.b)
    def get_confm(self):
        modeling = ClassifierML(1)
        return modeling.Class_confussionmatrix(self.b)


    def handle_reject_propose(self, message):
        """
        """
        super(CompContNet2, self).handle_reject_propose(message)

        display_message(self.agent.aid.name,
                        'REJECT_PROPOSAL message received')

    def handle_accept_propose(self, message):
        """
        """
        super(CompContNet2, self).handle_accept_propose(message)

        display_message(self.agent.aid.name,
                        'ACCEPT_PROPOSE message received')
        answer = message.create_reply()
        answer.set_performative(ACLMessage.INFORM)
        answer.set_content('OK')
        self.agent.send(answer)

#create master agent
class AgentInitiator(Agent):

    def __init__(self, aid, participants):
        super(AgentInitiator, self).__init__(aid=aid, debug=False)

        message = ACLMessage(ACLMessage.CFP)
        message.set_protocol(ACLMessage.FIPA_CONTRACT_NET_PROTOCOL)
        message.set_content('60.0')

        for participant in participants:
            message.add_receiver(AID(name=participant))

        self.call_later(8.0, self.launch_contract_net_protocol,message)  #Wait for all agent ready

    def launch_contract_net_protocol(self, message):
        comp = CompContNet1(self, message)
        self.behaviours.append(comp)
        comp.on_start()



#create slave agent
class AgentParticipant(Agent):

    def __init__(self, aid, pot_disp, Mlid,stategy):
        super(AgentParticipant, self).__init__(aid=aid, debug=False)

        self.pot_disp = pot_disp
        self.Mlid = Mlid
        self.stategy=stategy

        comp = CompContNet2(self, self.Mlid,self.stategy)

        self.behaviours.append(comp)


if __name__ == "__main__":
    #ignore some warning
    warnings.filterwarnings(action='ignore', category=RuntimeWarning)
    warnings.filterwarnings(action='ignore', category=FutureWarning)
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    agents_per_process = 7 # number of model in this MAS
    c = 0
    agents = list()
    participants = list()

    #Choose type of evaluation value for make weight
    stategy=random.randint(1,4)
    #stategy=4


    labels = ['KNN', 'Naivebayes', 'Bagging', 'Boosting', 'Decision tree', 'RandomForest', 'LDA']
    #Assign Id and create slave agent
    for i in range(agents_per_process):
        port = int(argv[1]) + c #for port setting(it's opptional)
        k = 900
        agent_name =labels[i]
        participants.append(agent_name)
        agente_ML = AgentParticipant(AID(name=agent_name), uniform(100.0, 500.0), i,stategy)

        agents.append(agente_ML)

        c += 100
    # Create Master agent
    agent_name = 'Master_agent{}@localhost:{}'.format(port, port)
    agente_init_1 = AgentInitiator(AID(name=agent_name), participants)
    agents.append(agente_init_1)

    #Load dataset dataframe to memory and split them
    datasetc.datagiv()
    #start Multiagent system
    start_loop(agents)
