from typing import List

OPEN_STATE = 'open'
CLOSED_STATE = 'completed'
ABANDONED_STATE = 'abandoned'
IN_PROGRESS_STATE = 'in_progress'
STATES = [OPEN_STATE, CLOSED_STATE, ABANDONED_STATE, IN_PROGRESS_STATE]

class Task:
    id: str
    goal: str
    parent: "Task | None"
    subtasks: List["Task"]

    def __init__(self, parent: "Task | None", goal: str, state: str=OPEN_STATE, subtasks: List = []):
        if parent is None:
            self.id = '0'
        else:
            self.id = parent.id + '.' + str(len(parent.subtasks))
        self.parent = parent
        self.goal = goal
        self.subtasks = subtasks
        self.state = OPEN_STATE

    def to_dict(self):
        return {
            'id': self.id,
            'goal': self.goal,
            'state': self.state,
            'subtasks': [t.to_dict() for t in self.subtasks]
        }

    def set_state(self, state):
        if state not in STATES:
            raise ValueError('Invalid state:' + state)
        self.state = state
        if state == CLOSED_STATE or state == ABANDONED_STATE:
            for subtask in self.subtasks:
                subtask.set_state(CLOSED_STATE)
        elif state == IN_PROGRESS_STATE:
            if self.parent is not None:
                self.parent.set_state(state)

    def get_current_task(self) -> "Task | None":
        for subtask in self.subtasks:
            if subtask.state == IN_PROGRESS_STATE:
                return subtask.get_current_task()
        if self.state == IN_PROGRESS_STATE:
            return self
        return None

class Plan:
    main_goal: str
    task: Task

    def __init__(self, task: str):
        self.main_goal = task
        self.task = Task(parent=None, goal=task, subtasks=[])

    def get_task_by_id(self, id: str) -> Task:
        try:
            parts = [int(p) for p in id.split('.')]
        except ValueError:
            raise ValueError('Invalid task id, non-integer:' + id)
        if parts[0] != 0:
            raise ValueError('Invalid task id, must start with 0:' + id)
        parts = parts[1:]
        task = self.task
        for part in parts:
            if part >= len(task.subtasks):
                raise ValueError('Task does not exist:' + id)
            task = task.subtasks[part]
        return task

    def add_subtask(self, parent_id: str, goal: str):
        parent = self.get_task_by_id(parent_id)
        id = parent.id + '.' + str(len(parent.subtasks))
        child = Task(parent=parent, goal=goal, subtasks=[])
        parent.subtasks.append(child)

    def set_subtask_state(self, id: str, state: str):
        task = self.get_task_by_id(id)
        task.set_state(state)

    def get_current_task(self):
        return self.task.get_current_task()

