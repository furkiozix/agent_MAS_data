# -*- coding: utf-8 -*-
from pade.misc.utility import display_message, start_loop
from pade.core.agent import Agent
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import FipaContractNetProtocol
from sys import argv
from random import uniform
import random
from collections import Counter

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from pandas import read_csv
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict

import numpy as np
import matplotlib.pyplot as plt
import warnings

class datasetc:
    X=np.array([0])
    Y=np.array([0])
    X_test=np.array([0])
    X_train=np.array([0])
    Y_train = np.array([0])
    Y_test = np.array([0])
    def __init__(self):
        print("nothingtodo")
    @staticmethod
    def datagiv():
        # make a model by K-NN
        filename = 'csvfile/pima-indians-diabetes.csv'
        #names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
        dataframe = read_csv(filename)

        array = dataframe.values  # Convert dataframe to numpy array
        # Model Making
        datasetc.X = array[:, 0:8]
        datasetc.Y = array[:, 8]
        test_size = 0.5  # Split test 33.33%, training 63.33%
        datasetc.X_train,datasetc.X_test,datasetc.Y_train,datasetc.Y_test = train_test_split(datasetc.X,datasetc.Y, test_size=test_size, random_state=7)




class ClassifierML:



    def __init__(self, MLid):
        self.MLid = MLid

    def knnpima(self):


        model = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
        model.fit(datasetc.X_train, datasetc.Y_train)

        # model Evaluation

        # Accuracy value
        result = model.score(datasetc.X_test, datasetc.Y_test)
        thismopre=model.predict_proba(datasetc.X)
        tavg=np.average(thismopre,axis=0,weights=None)
        # print("Accuracy %.2f%%" % (result * 100.0))

        # fpr and tpr
        Y_scores = model.predict_proba(datasetc.X_test)
        fpr, tpr, threshold = roc_curve(datasetc.Y_test, Y_scores[:, 1])
        roc_auc = auc(fpr, tpr)



        # precision value
        #precision, recall, thresholds = precision_recall_curve(datasetc.Y_test, Y_scores[:, 1])

        #yhat = model.predict(datasetc.X_test)
        # f1 score
        
        # print(classification_report(Y_test, yhat))

        # cofustion matrix

        #cm = confusion_matrix(datasetc.Y_test, yhat)
        # print(cm)
        scores = cross_val_score(model, datasetc.X, datasetc.Y, cv=5)
        #t_pred = cross_val_predict(model, datasetc.X, datasetc.Y, cv=3, method='predict_proba')
        pack = [tavg[0], tavg[1], fpr, tpr, roc_auc]
        return pack



    def nbayiris(self):
        # Model Making


        # training a Naive Bayes classifier

        gnb = GaussianNB()
        y_pred = gnb.fit(datasetc.X, datasetc.Y).predict(datasetc.X)

        # cofustion matrix
        cm = confusion_matrix(datasetc.Y, y_pred)
        # print(cm)
        thismopre = gnb.predict_proba(datasetc.X)
        tavg = np.average(thismopre, axis=0, weights=None)

        # Accuracy value
        result = gnb.score(datasetc.X_test, datasetc.Y_test)
        # print("Accuracy %.2f%%" % (result * 100.0))

        # fpr and tpr

        fpr, tpr, threshold = roc_curve(datasetc.Y, y_pred,pos_label=1)
        roc_auc = auc(fpr, tpr)




        # precision value
        #precision, recall, thresholds = precision_recall_curve(datasetc.Y, y_pred, pos_label=3)

        # f1 score

        # print(classification_report(y, y_pred))
        scores = cross_val_score(gnb,datasetc.X,datasetc.Y, cv=5)
        pack = [tavg[0], tavg[1], fpr, tpr, roc_auc]
        return pack


    def decitree(self):

        clf2 = tree.DecisionTreeClassifier()
        clf2 = clf2.fit(datasetc.X_train, datasetc.Y_train)

        y_pred_en = clf2.predict(datasetc.X_test)

        # cofustion matrix
        cm = confusion_matrix(datasetc.Y_test, y_pred_en)
        # print(cm)

        # Accuracy value
        result = clf2.score(datasetc.X_test, datasetc.Y_test)
        thismopre = clf2.predict_proba(datasetc.X)
        tavg= np.average(thismopre, axis=0, weights=None)
        # print("Accuracy:", metrics.accuracy_score(y_test, y_pred_en) * 100)
        results = metrics.accuracy_score(datasetc.Y_test, y_pred_en)

        # fpr and tpr

        fpr, tpr, threshold = roc_curve(datasetc.Y_test, y_pred_en)
        roc_auc = auc(fpr, tpr)
        packroc = [fpr, tpr, roc_auc]


        # precision value
        precision, recall, thresholds = precision_recall_curve(datasetc.Y_test, y_pred_en, pos_label=2)

        # f1 score

        # print(classification_report(y_test, y_pred_en))
        scores = cross_val_score(clf2, datasetc.X, datasetc.Y, cv=5)
        #t_pred = cross_val_predict(clf2, datasetc.X, datasetc.Y, cv=3, method='predict_proba')
        pack = [tavg[0], tavg[1], fpr, tpr, roc_auc]
        return pack


    def ranfores(self):

        Random_Forest_model = RandomForestClassifier(n_estimators=100)
        model = Random_Forest_model.fit(datasetc.X_train, datasetc.Y_train)
        thismopre = model.predict_proba(datasetc.X)
        tavg = np.average(thismopre, axis=0, weights=None)

        y_pred_en = model.predict(datasetc.X_test)

        # cofustion matrix
        cm = confusion_matrix(datasetc.Y_test, y_pred_en)
        # print(cm)

        # Accuracy value
        result = model.score(datasetc.X_test, datasetc.Y_test)
        # print("Accuracy %.2f%%" % (result * 100.0))

        # fpr and tpr

        fpr, tpr, threshold = roc_curve(datasetc.Y_test, y_pred_en,pos_label=1)
        roc_auc = auc(fpr, tpr)



        # precision value
        precision, recall, thresholds = precision_recall_curve(datasetc.Y_test, y_pred_en, pos_label=3)

        # f1 score

        # print(classification_report(y_test, y_pred_en))
        scores = cross_val_score(model, datasetc.X, datasetc.Y, cv=5)
        t_pred = cross_val_predict(model, datasetc.X, datasetc.Y, cv=3, method='predict_proba')
        pack = [tavg[0], tavg[1], fpr, tpr, roc_auc]
        return pack

    def boosmodel(self):
        # Model Making
        # LDA = LinearDiscriminantAnalysis( n_components=1)

        # X_train = LDA.fit_transform(x_train, y_train)
        # X_test = LDA.transform(x_test)
        seed = 3
        num_trees = 10
        kfold = model_selection.KFold(n_splits=20, random_state=seed)
        model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
        model = model.fit(datasetc.X_train,datasetc. Y_train)
        thismopre = model.predict_proba(datasetc.X)
        tavg = np.average(thismopre, axis=0, weights=None)

        # Accuracy value
        results = model_selection.cross_val_score(model, datasetc.X, datasetc.Y, cv=kfold)
        result = (results.mean())


        y_pred_en = model.predict(datasetc.X_test)

        # cofustion matrix
        cm = confusion_matrix(datasetc.Y_test, y_pred_en)
        # print(cm)

        # fpr and tpr

        fpr, tpr, threshold = roc_curve(datasetc.Y_test, y_pred_en)
        roc_auc = auc(fpr, tpr)
        packroc = [fpr, tpr, roc_auc]



        # precision value
        precision, recall, thresholds = precision_recall_curve(datasetc.Y_test, y_pred_en)

        # f1 score

        # print(classification_report(y_test, y_pred_en))

        t_pred = cross_val_predict(model, datasetc.X, datasetc.Y, cv=3, method='predict_proba')
        pack=[tavg[0],tavg[1],fpr,tpr,roc_auc]
        return pack

    def bagging(self):
        # Model Making


        seed = 3

        cart = DecisionTreeClassifier()
        num_trees = 100
        model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed).fit(datasetc.X_train,datasetc.Y_train)
        thismopre = model.predict_proba(datasetc.X)
        tavg = np.average(thismopre, axis=0, weights=None)
        # accuracy
        results = model_selection.cross_val_score(model, datasetc.X_train, datasetc.Y_train,cv=5)
        result = results.mean()

        y_pred_en = model.predict(datasetc.X_test)
        # cofustion matrix
        cm = confusion_matrix(datasetc.Y_test, y_pred_en)
        # print(cm)

        # fpr and tpr

        fpr, tpr, threshold = roc_curve(datasetc.Y_test, y_pred_en)
        roc_auc = auc(fpr, tpr)
        packroc = [fpr, tpr, roc_auc]


        # precision value
        precision, recall, thresholds = precision_recall_curve(datasetc.Y_test, y_pred_en)

        # f1 score

        # print(classification_report(y_test, y_pred_en))

        scores = cross_val_score(model, datasetc.X, datasetc.Y, cv=3)
        #t_pred = cross_val_predict(model, datasetc.X, datasetc.Y, cv=3, method='predict_proba')
        packroc = [fpr, tpr, roc_auc]
        pack = [tavg[0], tavg[1], fpr, tpr, roc_auc]
        return pack

    def LDA(self):


        # 1. Instantiate the method and fit_transform the algotithm
        LDA = LinearDiscriminantAnalysis(n_components=1)

        X_train = LDA.fit_transform(datasetc.X_train, datasetc.Y_train)
        X_test = LDA.transform(datasetc.X_test)

        classifier = RandomForestClassifier(max_depth=2, random_state=0)

        classifier.fit(X_train, datasetc.Y_train)
        thismopre = classifier.predict_proba(X_train)
        tavg = np.average(thismopre, axis=0, weights=None)

        y_pred = classifier.predict(X_test)

        # accuracy
        cm = confusion_matrix(datasetc.Y_test, y_pred)
        # print(cm)
        result = accuracy_score(datasetc.Y_test, y_pred)
        # print('Accuracy' + str(accuracy_score(y_test, y_pred)))

        # fpr and tpr


        fpr, tpr, threshold = roc_curve(datasetc.Y_test, y_pred,pos_label=1)
        roc_auc = auc(fpr, tpr)


        # precision value
        precision, recall, thresholds = precision_recall_curve(datasetc.Y_test, y_pred)

        # f1 score

        # print(classification_report(y_test, y_pred))
        scores = cross_val_score( classifier, datasetc.X, datasetc.Y, cv=5)
        #t_pred = cross_val_predict( classifier, datasetc.X, datasetc.Y, cv=3, method='predict_proba')
        pack = [tavg[0], tavg[1], fpr, tpr, roc_auc]
        return pack
    def majorclassifire(self):
        clf = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
        clf2 = GaussianNB()
        clf3 = tree.DecisionTreeClassifier()
        clf4 = RandomForestClassifier(n_estimators=100, random_state=1)
        clf5 = AdaBoostClassifier(n_estimators=10, random_state=3)
        cart = DecisionTreeClassifier()
        clf6 = BaggingClassifier(base_estimator=cart, n_estimators=100, random_state=3)

        eclf = VotingClassifier(estimators=[('knn', clf), ('gnb', clf2),('dct', clf3), ('rf', clf4),('abo', clf5), ('bcf', clf6)], voting='soft').fit(datasetc.X_train,datasetc.Y_train)
        thismopre = eclf.predict_proba(datasetc.X_train)
        tavg = np.average(thismopre, axis=0, weights=None)
        y_pred = eclf.predict(datasetc.X_test)
        fpr, tpr, threshold = roc_curve(datasetc.Y_test, y_pred, pos_label=1)
        roc_auc = auc(fpr, tpr)
        pack = [tavg[0], tavg[1], fpr, tpr, roc_auc]
        return pack





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

    def handle_all_proposes(self, proposes):
        """
        """
        modeling = ClassifierML(1)
        super(CompContNet1, self).handle_all_proposes(proposes)

        best_proposer = None
        higher_power = 0.0
        other_proposers = list()
        display_message(self.agent.aid.name, 'Model Making...')
        b = modeling.majorclassifire()
        b2=np.array([b[0],b[1]])
        display_message(self.agent.aid.name, 'Analyzing.......')

        i = 0

        # logic to select proposals by the higher available power.
        s=np.array([],dtype=float)

        labels = ['KNN', 'Naivebayes', 'Decision tree','RandomForest','Boosting','Bagging','LDA','Ensemble']
        for message in proposes:
            content = message.content
            power = np.array(content)
            i += 1
            power2=np.array([power[0],power[1]])
            display_message(self.agent.aid.name,
                            'Analyzing Accuracy {i}'.format(i=i))
            #display_message(self.agent.aid.name,'Accuracy Offered: {pot}'.format(pot=power))

            if i<2 :
                s=np.vstack([s,power2])if s else power2
            else:
                s = np.vstack([s,power2])
            plt.plot(power[2], power[3], label='%s (auc = %0.2f)' % (labels[i-1], power[4]))
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            plt.grid()
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            other_proposers.append(message.sender)

            if i==7 :
                s = np.vstack([s,b2])
                plt.plot(b[2], b[3], label='%s (auc = %0.2f)' % (labels[i], b[4]))
                plt.legend(loc='lower right')
                plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=2)
                plt.xlim([-0.1, 1.1])
                plt.ylim([-0.1, 1.1])
                plt.grid()
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')



        s_prob=np.average(s,axis=0,weights=None)


        display_message(self.agent.aid.name,
                        'The averange of all predict probability is : {pot} '.format(
                            pot=s_prob))

        plt.show()
        display_message(self.agent.aid.name,
                            'Sending ACCEPT_PROPOSAL answers...')
        answer = ACLMessage(ACLMessage.ACCEPT_PROPOSAL)
        answer.set_protocol(ACLMessage.FIPA_CONTRACT_NET_PROTOCOL)
        answer.set_content('')
        for agent in other_proposers:
             answer.add_receiver(agent)

        self.agent.send(answer)


    def handle_inform(self, message):
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


class CompContNet2(FipaContractNetProtocol):

    '''CompContNet2

       FIPA-ContractNet Participant Behaviour that runs when an agent
       receives a CFP message. A proposal is sent and if it is selected,
       the restrictions are analized to enable the restoration.'''

    def __init__(self,agent,mlid):
        super(CompContNet2, self).__init__(agent=agent,
                                           message=None,
                                           is_initiator=False)
        self.mlid=mlid

    def handle_cfp(self, message):
        """
        """
        super(CompContNet2, self).handle_cfp(message)
        self.message = message

      
        display_message(self.agent.aid.name, 'CFP message received')

        answer = self.message.create_reply()
        answer.set_performative(ACLMessage.PROPOSE)
        answer.set_content(self.modelmaker(self.mlid))
        self.agent.send(answer)

    def modelmaker(self,mlid):

        modeling = ClassifierML(1)
        if  mlid == 0:
            display_message(self.agent.aid.name, 'This is K-nn model')
            b=modeling.knnpima()

        elif mlid == 1:
            display_message(self.agent.aid.name, 'This is Naive-Bayes model')
            b =modeling.nbayiris()

        elif  mlid == 2:
            display_message(self.agent.aid.name, 'This is Bagging algorithm model')
            b =  modeling.bagging()

        elif  mlid == 3:
            display_message(self.agent.aid.name, 'This is Boosting algorithm model')
            b =  modeling.boosmodel()

        elif  mlid ==4:
            display_message(self.agent.aid.name, 'This is Decision tree model')
            b = modeling.decitree()

        elif  mlid ==5 :
            display_message(self.agent.aid.name, 'This is Random forest model')
            b =  modeling.ranfores()

        else :
            display_message(self.agent.aid.name, 'This is  model that use LDA')
            b = modeling.LDA()
        return b



    def handle_reject_propose(self, message):
        """
        """
        super(CompContNet2, self).handle_reject_propose(message)

        display_message(self.agent.aid.name,
                        'REJECT_PROPOSAL message received')

    def handle_accept_propose(self, message):
        super(CompContNet2, self).handle_accept_propose(message)

        display_message(self.agent.aid.name,
                        'ACCEPT_PROPOSE message received')
        answer = message.create_reply()
        answer.set_performative(ACLMessage.INFORM)
        answer.set_content('OK')
        self.agent.send(answer)


class AgentInitiator(Agent):

    def __init__(self, aid, participants):
        super(AgentInitiator, self).__init__(aid=aid, debug=False)

        message = ACLMessage(ACLMessage.CFP)
        message.set_protocol(ACLMessage.FIPA_CONTRACT_NET_PROTOCOL)
        message.set_content('60.0')

        for participant in participants:
            message.add_receiver(AID(name=participant))

        self.call_later(8.0, self.launch_contract_net_protocol, message)

    def launch_contract_net_protocol(self, message):
        comp = CompContNet1(self, message)
        self.behaviours.append(comp)
        comp.on_start()


class AgentParticipant(Agent):

    def __init__(self, aid, pot_disp,Mlid):
        super(AgentParticipant, self).__init__(aid=aid, debug=False)

        self.pot_disp = pot_disp
        self.Mlid = Mlid

        comp = CompContNet2(self,self.Mlid)

        self.behaviours.append(comp)


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore',category=RuntimeWarning)
    warnings.filterwarnings(action='ignore', category=FutureWarning)
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)

    agents_per_process = 7
    c = 0
    agents = list()
    participants = list()
    for i in range(agents_per_process):
        port = int(argv[1]) + c        
        k = 900
        agent_name = 'agent_participant_{}@localhost:{}'.format(port - k, port - k)
        participants.append(agent_name)
        agente_ML = AgentParticipant(AID(name=agent_name), uniform(100.0, 500.0),i)

        agents.append(agente_ML)

        c += 100


    agent_name = 'agent_initiator_{}@localhost:{}'.format(port, port)
    agente_init_1 = AgentInitiator(AID(name=agent_name), participants)
    agents.append(agente_init_1)
    datasetc.datagiv()
    start_loop(agents)
