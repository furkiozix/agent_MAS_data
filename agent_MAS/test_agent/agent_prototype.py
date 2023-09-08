# -*- coding: utf-8 -*-
from pade.misc.utility import display_message, start_loop
from pade.core.agent import Agent
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import FipaContractNetProtocol
from sys import argv
from random import uniform
import random

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier
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



class ClassifierML:

    def __init__(self, MLid):
        self.MLid = MLid

    def knnpima(self):
        # make a model by K-NN
        filename = 'csvfile/pe_section_headers.csv'
        names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
        dataframe = read_csv(filename, names=names)

        array = dataframe.values  # Convert dataframe to numpy array
        # Model Making
        X = array[:, 0:8]
        Y = array[:, 8]
        test_size = 0.33  # Split test 33.33%, training 63.33%

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=7)

        model = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
        model.fit(X_train, Y_train)

        # model Evaluation

        # Accuracy value
        result = model.score(X_test, Y_test)
        # print("Accuracy %.2f%%" % (result * 100.0))

        # fpr and tpr
        Y_scores = model.predict_proba(X_test)
        fpr, tpr, threshold = roc_curve(Y_test, Y_scores[:, 1])
        roc_auc = auc(fpr, tpr)

        # precision value
        precision, recall, thresholds = precision_recall_curve(Y_test, Y_scores[:, 1])

        yhat = model.predict(X_test)
        # f1 score
        # print(classification_report(Y_test, yhat))

        # cofustion matrix

        cm = confusion_matrix(Y_test, yhat)
        # print(cm)
        scores = cross_val_score(model, X, Y, cv=5)
        t_pred = cross_val_predict(model, X, Y, cv=3, method='predict_proba')
        return scores.mean()

    def nbayiris(self):
        data = read_csv('csvfile/iris.csv')
        data.variety.replace(['Setosa', 'Versicolor', 'Virginica'], [1, 2, 3], inplace=True)
        array = data.values
        # X -> features, y -> label
        X = array[:, 0:4]
        y = array[:, 4]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

        # dividing X, y into train and test data

        # training a Naive Bayes classifier

        gnb = GaussianNB()
        y_pred = gnb.fit(X, y).predict(X)

        # cofustion matrix
        cm = confusion_matrix(y, y_pred)
        # print(cm)

        # Accuracy value
        result = gnb.score(X_test, y_test)
        # print("Accuracy %.2f%%" % (result * 100.0))

        # fpr and tpr

        fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=3)
        roc_auc = auc(fpr, tpr)

        # precision value
        precision, recall, thresholds = precision_recall_curve(y, y_pred, pos_label=3)

        # f1 score

        # print(classification_report(y, y_pred))
        scores = cross_val_score(gnb, X, y, cv=5)
        t_pred = cross_val_predict(gnb, X, y, cv=3, method='predict_proba')
        return scores.mean()


    def decitree(self):
        df = read_csv('csvfile/abalone.csv')

        df = df.replace('M', 0)
        df = df.replace('F', 1)
        df = df.replace('I', 2)

        data = df.values
        X = data[:, 1:-1]
        Y = data[:, 0]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        clf2 = tree.DecisionTreeClassifier()
        clf2 = clf2.fit(X_train, y_train)

        y_pred_en = clf2.predict(X_test)

        # cofustion matrix
        cm = confusion_matrix(y_test, y_pred_en)
        # print(cm)

        # Accuracy value
        result = clf2.score(X_test, y_test)
        # print("Accuracy:", metrics.accuracy_score(y_test, y_pred_en) * 100)
        results = metrics.accuracy_score(y_test, y_pred_en)

        # fpr and tpr

        fpr, tpr, threshold = roc_curve(y_test, y_pred_en, pos_label=2)
        roc_auc = auc(fpr, tpr)

        # precision value
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_en, pos_label=2)

        # f1 score

        # print(classification_report(y_test, y_pred_en))
        scores = cross_val_score(clf2, X, Y, cv=5)
        t_pred = cross_val_predict(clf2, X, Y, cv=3, method='predict_proba')
        return scores.mean()


    def ranfores(self):
        data = read_csv('csvfile/iris.csv')
        data.variety.replace(['Setosa', 'Versicolor', 'Virginica'], [1, 2, 3], inplace=True)
        array = data.values
        # X -> features, y -> label
        X = array[:, 0:4]
        y = array[:, 4]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        Random_Forest_model = RandomForestClassifier(n_estimators=100)
        model = Random_Forest_model.fit(X_train, y_train)

        y_pred_en = model.predict(X_test)

        # cofustion matrix
        cm = confusion_matrix(y_test, y_pred_en)
        # print(cm)

        # Accuracy value
        result = model.score(X_test, y_test)
        # print("Accuracy %.2f%%" % (result * 100.0))

        # fpr and tpr

        fpr, tpr, threshold = roc_curve(y_test, y_pred_en, pos_label=3)
        roc_auc = auc(fpr, tpr)

        # precision value
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_en, pos_label=3)

        # f1 score

        # print(classification_report(y_test, y_pred_en))
        scores = cross_val_score(model, X, y, cv=5)
        t_pred = cross_val_predict(model, X, y, cv=3, method='predict_proba')
        return scores.mean()

    def boosmodel(self):
        data = read_csv("csvfile/bank.csv")

        le = LabelEncoder()
        for col in data.columns:
            if data[col].dtypes == 'object':  ####### when column's data type is equal to object
                data[col] = le.fit_transform(data[col])  ###### fit_transform is used for conversion

        array = data.values
        X = array[:, 0:16]
        Y = array[:, 16]

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        # LDA = LinearDiscriminantAnalysis( n_components=1)

        # X_train = LDA.fit_transform(x_train, y_train)
        # X_test = LDA.transform(x_test)
        seed = 3
        num_trees = 10
        kfold = model_selection.KFold(n_splits=20, random_state=seed)
        model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
        model = model.fit(x_train, y_train)

        # Accuracy value
        results = model_selection.cross_val_score(model, X, Y, cv=kfold)
        result = (results.mean())

        y_pred_en = model.predict(x_test)

        # cofustion matrix
        cm = confusion_matrix(y_test, y_pred_en)
        # print(cm)

        # fpr and tpr

        fpr, tpr, threshold = roc_curve(y_test, y_pred_en, pos_label=2)
        roc_auc = auc(fpr, tpr)

        # precision value
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_en, pos_label=2)

        # f1 score

        # print(classification_report(y_test, y_pred_en))

        t_pred = cross_val_predict(model, X, Y, cv=3, method='predict_proba')
        return result

    def bagging(self):
        data = read_csv("csvfile/bank.csv")

        le = LabelEncoder()
        for col in data.columns:
            if data[col].dtypes == 'object':  ####### when column's data type is equal to object
                data[col] = le.fit_transform(data[col])  ###### fit_transform is used for conversion

        array = data.values
        X = array[:, 0:16]
        Y = array[:, 16]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        LDA = LinearDiscriminantAnalysis(n_components=1)

        X_train = LDA.fit_transform(X_train, y_train)
        X_test = LDA.transform(X_test)

        seed = 3

        cart = DecisionTreeClassifier()
        num_trees = 100
        model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed).fit(X_train, y_train)

        # accuracy
        results = model_selection.cross_val_score(model, X_train, y_train,cv=5)
        result = results.mean()

        y_pred_en = model.predict(X_test)
        # cofustion matrix
        cm = confusion_matrix(y_test, y_pred_en)
        # print(cm)

        # fpr and tpr

        fpr, tpr, threshold = roc_curve(y_test, y_pred_en)
        roc_auc = auc(fpr, tpr)

        # precision value
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_en)

        # f1 score

        # print(classification_report(y_test, y_pred_en))

        scores = cross_val_score(model, X, Y, cv=3)
        t_pred = cross_val_predict(model, X, Y, cv=3, method='predict_proba')
        return scores.mean()

    def LDA(self):
        data = read_csv("csvfile/bank.csv")

        le = LabelEncoder()
        for col in data.columns:
            if data[col].dtypes == 'object':  ####### when column's data type is equal to object
                data[col] = le.fit_transform(data[col])  ###### fit_transform is used for conversion

        # print(data.shape)

        array = data.values
        X = array[:, 0:16]
        Y = array[:, 16]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        # 1. Instantiate the method and fit_transform the algotithm
        LDA = LinearDiscriminantAnalysis(n_components=1)

        X_train = LDA.fit_transform(X_train, y_train)
        X_test = LDA.transform(X_test)

        classifier = RandomForestClassifier(max_depth=2, random_state=0)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        # accuracy
        cm = confusion_matrix(y_test, y_pred)
        # print(cm)
        result = accuracy_score(y_test, y_pred)
        # print('Accuracy' + str(accuracy_score(y_test, y_pred)))

        # fpr and tpr

        fpr, tpr, threshold = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        # precision value
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

        # f1 score

        # print(classification_report(y_test, y_pred))
        scores = cross_val_score( classifier, X, Y, cv=5)
        t_pred = cross_val_predict( classifier, X, Y, cv=3, method='predict_proba')
        return scores.mean()



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

        super(CompContNet1, self).handle_all_proposes(proposes)

        best_proposer = None
        higher_power = 0.0
        other_proposers = list()
        display_message(self.agent.aid.name, 'Analyzing accuracy...')

        i = 1

        # logic to select proposals by the higher available power.
        for message in proposes:
            content = message.content
            power = float(content)
            display_message(self.agent.aid.name,
                            'Analyzing proposal {i}'.format(i=i))
            display_message(self.agent.aid.name,
                            'Power Offered: {pot}'.format(pot=power))
            i += 1

            if power > higher_power:
                if best_proposer is not None:
                    other_proposers.append(best_proposer)

                higher_power = power
                best_proposer = message.sender
            else:
                other_proposers.append(message.sender)
        display_message(self.agent.aid.name,
                        'The best accuracy was: {pot} %'.format(
                            pot=higher_power))

        if other_proposers != []:
            display_message(self.agent.aid.name,
                            'Sending REJECT_PROPOSAL answers...')
            answer = ACLMessage(ACLMessage.REJECT_PROPOSAL)
            answer.set_protocol(ACLMessage.FIPA_CONTRACT_NET_PROTOCOL)
            answer.set_content('')
            for agent in other_proposers:
                answer.add_receiver(agent)

            self.agent.send(answer)

        if best_proposer is not None:
            display_message(self.agent.aid.name,
                            'Sending ACCEPT_PROPOSAL answer...')

            answer = ACLMessage(ACLMessage.ACCEPT_PROPOSAL)
            answer.set_protocol(ACLMessage.FIPA_CONTRACT_NET_PROTOCOL)
            answer.set_content('OK')
            answer.add_receiver(best_proposer)
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


class CompContNet2(FipaContractNetProtocol):
    '''CompContNet2

       FIPA-ContractNet Participant Behaviour that runs when an agent
       receives a CFP message. A proposal is sent and if it is selected,
       the restrictions are analized to enable the restoration.'''


    def ranafort(self):
        a = random.randint(1, 7)
        return  a
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
        """
        """
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
    agents_per_process = 7
    c = 0
    agents = list()
    participants = list()
    labels = ['KNN', 'Naivebayes', 'Bagging', 'Boosting','Decision tree', 'RandomForest', 'LDA']
    for i in range(agents_per_process):
        port = int(argv[1]) + c        
        k = 900
        agent_name = labels[i]
        participants.append(agent_name)
        agente_ML = AgentParticipant(AID(name=agent_name), uniform(100.0, 500.0),i)

        agents.append(agente_ML)

        c += 100

    agent_name = 'Master agent'
    agente_init_1 = AgentInitiator(AID(name=agent_name), participants)
    agents.append(agente_init_1)
    
    start_loop(agents)
