import random
import copy
import numpy as np

from skExSTraCS.CodeFragment import CodeFragment
from skExSTraCS.Condition import Condition

class Classifier:
    def __init__(self,model):
        self.specifiedAttList = []
        self.condition = []
        self.phenotype = None

        self.fitness = model.init_fitness
        self.accuracy = 0
        self.numerosity = 1
        self.aveMatchSetSize = None
        self.deletionProb = None

        self.timeStampGA = None
        self.initTimeStamp = None
        self.epochComplete = False

        self.matchCount = 0
        self.correctCount = 0
        self.matchCover = 0
        self.correctCover = 0

    def initializeByCopy(self,toCopy,iterationCount):
        self.specifiedAttList = copy.deepcopy(toCopy.specifiedAttList)
        self.condition = copy.deepcopy(toCopy.condition)
        self.phenotype = copy.deepcopy(toCopy.phenotype)
        self.timeStampGA = iterationCount
        self.initTimeStamp = iterationCount
        self.aveMatchSetSize = copy.deepcopy(toCopy.aveMatchSetSize)
        self.fitness = toCopy.fitness
        self.accuracy = toCopy.accuracy

    def initializeByCovering(self,model,setSize,state,phenotype):
        self.timeStampGA = model.iterationCount
        self.initTimeStamp = model.iterationCount
        self.aveMatchSetSize = setSize
        self.phenotype = phenotype
        cond_len = len(state)
        toSpecify = int(model.p_spec * cond_len)

        condSpecified = random.sample(range(cond_len), toSpecify)
        for attRef in range(cond_len):
            condition = None
            if attRef in condSpecified:
                condition = self.buildMatch(model, state)  # Add classifierConditionElement
            if condition is None:
                condition = Condition()
            self.condition.append(condition)

    def buildMatch(self,model,state):
        attributes = list(range(0, model.env.formatData.numAttributes))
        for i in range(100):

            cf = CodeFragment.createCodeFragment(variables=attributes, level=model.level)

            result = CodeFragment.evaluate(cf, state)

            if result > 0.5:
                condition = Condition(cf)
            else:
                continue
            # Check if current rule already contains this expression
            if condition.expression in [cd.expression for cd in self.condition if not cd.is_dc]:
                continue

            return condition
        return None

    def updateEpochStatus(self,model):
        if not self.epochComplete and (model.iterationCount - self.initTimeStamp - 1) >= model.env.formatData.numTrainInstances:
            self.epochComplete = True

    def match(self, model, state):
        condition = self.condition

        for cd in condition:
            if cd.is_dc:
                continue
            result = CodeFragment.evaluate(cd.codeFragment, state)

            if result <= 0.5:
                return False

        return True

    def equals(self,cl):
        if cl.phenotype == self.phenotype and len(cl.getWorkingCondition()) == len(self.getWorkingCondition()):
            for cd in cl.condition:
                if cd.is_dc:
                    continue

                idx = self.findIndexByExpression(self, cd.expression)
                if idx is None or str(cd) != str(self.condition[idx]):
                    return False
            return True

        return False

    def updateExperience(self):
        self.matchCount += 1
        if self.epochComplete:  # Once epoch Completed, number of matches for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.matchCover += 1

    def updateMatchSetSize(self, model,matchSetSize):
        if self.matchCount < 1.0 / model.beta:
            self.aveMatchSetSize = (self.aveMatchSetSize * (self.matchCount-1)+ matchSetSize) / float(self.matchCount)
        else:
            self.aveMatchSetSize = self.aveMatchSetSize + model.beta * (matchSetSize - self.aveMatchSetSize)

    def updateCorrect(self):
        self.correctCount += 1
        if self.epochComplete: #Once epoch Completed, number of correct for a unique rule will not change, so do repeat calculation
            pass
        else:
            self.correctCover += 1

    def updateAccuracy(self):
        self.accuracy = self.correctCount / float(self.matchCount)

    def updateFitness(self,model):
        self.fitness = pow(self.accuracy, model.nu)

    def updateNumerosity(self, num):
        """ Alters the numberosity of the classifier.  Notice that num can be negative! """
        self.numerosity += num

    def isSubsumer(self, model):
        if self.matchCount > model.theta_sub and self.accuracy > model.acc_sub:
            return True
        return False

    def subsumes(self,model,cl):
        return cl.phenotype == self.phenotype and self.isSubsumer(model) and self.isMoreGeneral(model,cl)

    def isMoreGeneral(self,model, cl):
        if len(self.getWorkingCondition()) >= len(cl.getWorkingCondition()):
            return False

        for i in range(len(self.condition)):
            cd = self.condition[i]
            if cd.is_dc:
                continue
            clIndex = self.findIndexByExpression(cl, cd.expression)
            if clIndex is None:
                return False

        return True

    def updateTimeStamp(self, ts):
        """ Sets the time stamp of the classifier. """
        self.timeStampGA = ts

    def uniformCrossover(self,model,cl):
        changed = False
        x = random.randint(0, len(self.condition))
        y = random.randint(0, len(cl.condition))

        if x > y:
            x, y = y, x

        for i in range(x, y):
            if str(self.condition[i]) != str(cl.condition[i]):
                self.condition[i], cl.condition[i] = cl.condition[i], self.condition[i]
                changed = True
        return changed

    def specLimitFix(self, model, cl):
        """ Lowers classifier specificity to specificity limit. """
        if model.do_attribute_feedback:
            # Identify 'toRemove' attributes with lowest AT scores
            while len(cl.specifiedAttList) > model.rule_specificity_limit:
                minVal = model.AT.getTrackProb()[cl.specifiedAttList[0]]
                minAtt = cl.specifiedAttList[0]
                for j in cl.specifiedAttList:
                    if model.AT.getTrackProb()[j] < minVal:
                        minVal = model.AT.getTrackProb()[j]
                        minAtt = j
                i = cl.specifiedAttList.index(minAtt)  # reference to the position of the attribute in the rule representation
                cl.specifiedAttList.remove(minAtt)
                cl.condition.pop(i)  # buildMatch handles both discrete and continuous attributes

        else:
            # Randomly pick 'toRemove'attributes to be generalized
            toRemove = len(cl.specifiedAttList) - model.rule_specificity_limit
            genTarget = random.sample(cl.specifiedAttList, toRemove)
            for j in genTarget:
                i = cl.specifiedAttList.index(j)  # reference to the position of the attribute in the rule representation
                cl.specifiedAttList.remove(j)
                cl.condition.pop(i)  # buildMatch handles both discrete and continuous attributes

    def setAccuracy(self, acc):
        """ Sets the accuracy of the classifier """
        self.accuracy = acc

    def setFitness(self, fit):
        """  Sets the fitness of the classifier. """
        self.fitness = fit

    def mutation(self,model,state):
        changed = False
        # Mutate Condition
        for i in range(len(self.condition)):
            cd = self.condition[i]
            if random.random() < model.mu:
                # Mutation
                if cd.is_dc:
                    # dc -> cf
                    condition = self.buildMatch(model, state)
                    if condition is not None:
                        self.condition[i] = condition
                        changed = True
                else:
                    # cf -> dc
                    if random.random() > 0.5:
                        self.condition[i]= Condition()
                        changed = True
                    else: # cf -> new cf
                        condition = self.buildMatch(model, state)
                        if condition is not None:
                            self.condition[i] = condition
                            changed = True

        # Mutate Phenotype
        if model.env.formatData.discretePhenotype:
            nowChanged = self.discretePhenotypeMutation(model)

        if changed or nowChanged:
            return True


    def discretePhenotypeMutation(self, model):
        changed = False
        if random.random() < model.mu:
            phenotypeList = copy.deepcopy(model.env.formatData.phenotypeList)
            phenotypeList.remove(self.phenotype)
            newPhenotype = random.choice(phenotypeList)
            self.phenotype = newPhenotype
            changed = True
        return changed


    def selectGeneralizeRW(self,model,count):
        probList = []
        for attribute in self.specifiedAttList:
            probList.append(1/model.EK.scores[attribute])
        if sum(probList) == 0:
            probList = (np.array(probList) + 1).tolist()

        probList = np.array(probList)/sum(probList) #normalize
        return np.random.choice(self.specifiedAttList,count,replace=False,p=probList).tolist()

    # def selectGeneralizeRW(self,model,count):
    #     EKScoreSum = 0
    #     selectList = []
    #     currentCount = 0
    #     specAttList = copy.deepcopy(self.specifiedAttList)
    #     for i in self.specifiedAttList:
    #         # When generalizing, EK is inversely proportional to selection probability
    #         EKScoreSum += 1 / float(model.EK.scores[i] + 1)
    #
    #     while currentCount < count:
    #         choicePoint = random.random() * EKScoreSum
    #         i = 0
    #         sumScore = 1 / float(model.EK.scores[specAttList[i]] + 1)
    #         while choicePoint > sumScore:
    #             i = i + 1
    #             sumScore += 1 / float(model.EK.scores[specAttList[i]] + 1)
    #         selectList.append(specAttList[i])
    #         EKScoreSum -= 1 / float(model.EK.scores[specAttList[i]] + 1)
    #         specAttList.pop(i)
    #         currentCount += 1
    #     return selectList

    def selectSpecifyRW(self,model,count):
        pickList = list(range(model.env.formatData.numAttributes))
        for i in self.specifiedAttList:  # Make list with all non-specified attributes
            pickList.remove(i)

        probList = []
        for attribute in pickList:
            probList.append(model.EK.scores[attribute])
        if sum(probList) == 0:
            probList = (np.array(probList) + 1).tolist()
        probList = np.array(probList) / sum(probList)  # normalize
        return np.random.choice(pickList, count, replace=False, p=probList).tolist()

    # def selectSpecifyRW(self, model,count):
    #     """ EK applied to the selection of an attribute to specify for mutation. """
    #     pickList = list(range(model.env.formatData.numAttributes))
    #     for i in self.specifiedAttList:  # Make list with all non-specified attributes
    #         pickList.remove(i)
    #
    #     EKScoreSum = 0
    #     selectList = []
    #     currentCount = 0
    #
    #     for i in pickList:
    #         # When generalizing, EK is inversely proportional to selection probability
    #         EKScoreSum += model.EK.scores[i]
    #
    #     while currentCount < count:
    #         choicePoint = random.random() * EKScoreSum
    #         i = 0
    #         sumScore = model.EK.scores[pickList[i]]
    #         while choicePoint > sumScore:
    #             i = i + 1
    #             sumScore += model.EK.scores[pickList[i]]
    #         selectList.append(pickList[i])
    #         EKScoreSum -= model.EK.scores[pickList[i]]
    #         pickList.pop(i)
    #         currentCount += 1
    #     return selectList

    def mutateContinuousAttributes(self, model,useAT, j):
        # -------------------------------------------------------
        # MUTATE CONTINUOUS ATTRIBUTES
        # -------------------------------------------------------
        if useAT:
            if random.random() < model.AT.getTrackProb()[j]:  # High AT probability leads to higher chance of mutation (Dives ExSTraCS to explore new continuous ranges for important attributes)
                # Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
                attRange = float(model.env.formatData.attributeInfoContinuous[j][1]) - float(model.env.formatData.attributeInfoContinuous[j][0])
                i = self.specifiedAttList.index(j)  # reference to the position of the attribute in the rule representation
                mutateRange = random.random() * 0.5 * attRange
                if random.random() > 0.5:  # Mutate minimum
                    if random.random() > 0.5:  # Add
                        self.condition[i][0] += mutateRange
                    else:  # Subtract
                        self.condition[i][0] -= mutateRange
                else:  # Mutate maximum
                    if random.random() > 0.5:  # Add
                        self.condition[i][1] += mutateRange
                    else:  # Subtract
                        self.condition[i][1] -= mutateRange
                # Repair range - such that min specified first, and max second.
                self.condition[i].sort()
                changed = True
        elif random.random() > 0.5:
            # Mutate continuous range - based on Bacardit 2009 - Select one bound with uniform probability and add or subtract a randomly generated offset to bound, of size between 0 and 50% of att domain.
            attRange = float(model.env.formatData.attributeInfoContinuous[j][1]) - float(model.env.formatData.attributeInfoContinuous[j][0])
            i = self.specifiedAttList.index(j)  # reference to the position of the attribute in the rule representation
            mutateRange = random.random() * 0.5 * attRange
            if random.random() > 0.5:  # Mutate minimum
                if random.random() > 0.5:  # Add
                    self.condition[i][0] += mutateRange
                else:  # Subtract
                    self.condition[i][0] -= mutateRange
            else:  # Mutate maximum
                if random.random() > 0.5:  # Add
                    self.condition[i][1] += mutateRange
                else:  # Subtract
                    self.condition[i][1] -= mutateRange
            # Repair range - such that min specified first, and max second.
            self.condition[i].sort()
            changed = True
        else:
            pass


    def rangeCheck(self,model):
        """ Checks and prevents the scenario where a continuous attributes specified in a rule has a range that fully encloses the training set range for that attribute."""
        for attRef in self.specifiedAttList:
            if model.env.formatData.attributeInfoType[attRef]: #Attribute is Continuous
                trueMin = model.env.formatData.attributeInfoContinuous[attRef][0]
                trueMax = model.env.formatData.attributeInfoContinuous[attRef][1]
                i = self.specifiedAttList.index(attRef)
                valBuffer = (trueMax-trueMin)*0.1
                if self.condition[i][0] <= trueMin and self.condition[i][1] >= trueMax: # Rule range encloses entire training range
                    self.specifiedAttList.remove(attRef)
                    self.condition.pop(i)
                    return
                elif self.condition[i][0]+valBuffer < trueMin:
                    self.condition[i][0] = trueMin - valBuffer
                elif self.condition[i][1]- valBuffer > trueMax:
                    self.condition[i][1] = trueMin + valBuffer
                else:
                    pass

    def getDelProp(self, model, meanFitness):
        """  Returns the vote for deletion of the classifier. """
        if self.fitness / self.numerosity >= model.delta * meanFitness or self.matchCount < model.theta_del:
            deletionVote = self.aveMatchSetSize * self.numerosity
        elif self.fitness == 0.0:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (model.init_fitness / self.numerosity)
        else:
            deletionVote = self.aveMatchSetSize * self.numerosity * meanFitness / (self.fitness / self.numerosity)
        return deletionVote

    def findIndexByExpression(self, cl, expression):
        for i, cond in enumerate(cl.condition):
            if cond.expression == expression:
                return i
        return None

    def getWorkingCondition(self):
        return [cd for cd in self.condition if not cd.is_dc]

    def getDcCondition(self):
        return [cd for cd in self.condition if cd.is_dc]

    def getIndexByExpression(self, cl, expression):
        for i, cond in enumerate(cl.condition):
            if cond.expression == expression:
                return i

    def getWorkingConditionIndexs(self):
        return [i for i,cd in self.condition if not cd.is_dc]

    def getWorkingConditionIndex(self):
        indexs = self.getWorkingConditionIndexs()
        if indexs:
            return indexs[0]
        return None

    def getDcConditionIndexs(self):
        return [i for i,cd in self.condition if cd.is_dc]

    def getDcConditionIndex(self):
        indexs = self.getDcConditionIndexs()
        if indexs:
            return indexs[0]
        return None
