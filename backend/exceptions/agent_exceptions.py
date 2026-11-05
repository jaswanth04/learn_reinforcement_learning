class AgentException(Exception): 
    pass

class AgentTrainingException(AgentException):
    pass

class AgentInitializationException(AgentException):
    pass