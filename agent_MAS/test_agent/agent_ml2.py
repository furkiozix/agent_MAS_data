# -*- coding: utf-8 -*-
from pade.misc.utility import display_message, start_loop
from pade.core.agent import Agent
from pade.acl.aid import AID
from pade.acl.messages import ACLMessage
from pade.behaviours.protocols import FipaContractNetProtocol
from sys import argv
from random import uniform

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas import read_csv
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix




class ClassifierML:

    def __init__(self, MLid):
        self.MLid = MLid

    def knnpima(self):
        # make a model by K-NN
        filename = 'csvfile/pima-indians-diabetes.csv'
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

        return result

    def nbayiris(self):
        ata = pd.read_csv('csvfile/iris.csv')
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
        #print(cm)

        # Accuracy value
        result = gnb.score(X_test, y_test)
        #print("Accuracy %.2f%%" % (result * 100.0))

        # fpr and tpr

        fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=3)
        roc_auc = auc(fpr, tpr)

        # precision value
        precision, recall, thresholds = precision_recall_curve(y, y_pred, pos_label=3)

        # f1 score

        #print(classification_report(y, y_pred))


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
        display_message(self.agent.aid.name, 'Analyzing proposals...')

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
                        'The best proposal was: {pot} VA'.format(
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


    def __init__(self,Mlid, agent):
        super(CompContNet2, self).__init__(agent=agent,
                                           message=None,
                                           is_initiator=False)
        self.Mlid=Mlid



    def _handle_cfp(self, message):
        """
        """
        super(CompContNet2, self).handle_cfp(message)
        self.message = message

        display_message(self.agent.aid.name, 'CFP message received')


        modeling=ClassifierML(1)
        answer = self.message.create_reply()
        answer.set_performative(ACLMessage.PROPOSE)
        if self.Mlid == 'knn':
            answer.set_content(modeling.knnpima())
        else:
            answer.set_content(modeling.nbayiris())
        self.agent.send(answer)

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
    agents_per_process = 1
    c = 0
    agents = list()
    for i in range(agents_per_process):
        port = int(argv[1]) + c        
        k = 1001
        participants = list()

        agent_name = 'agent_participant_{}@localhost:{}'.format(port - k, port - k)
        participants.append(agent_name)
        agente_part_1 = AgentParticipant(AID(name=agent_name), uniform(100.0, 500.0),'knn')
        agents.append(agente_part_1)

        agent_name = 'agent_participant_{}@localhost:{}'.format(port + k, port + k)
        participants.append(agent_name)
        agente_part_2 = AgentParticipant(AID(name=agent_name), uniform(100.0, 500.0),'naive')
        agents.append(agente_part_2)

        agent_name = 'agent_initiator_{}@localhost:{}'.format(port, port)
        agente_init_1 = AgentInitiator(AID(name=agent_name), participants)
        agents.append(agente_init_1)

        c += 1000
    
    start_loop(agents)
