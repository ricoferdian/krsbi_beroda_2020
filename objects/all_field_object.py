

class AllFielOjects:
    def __init__(self):
        self.field_objects = []

    def add_object(self, field_object):
        self.field_objects.append(field_object)

    def empty_object(self):
        self.field_objects = []

    def get_objects(self):
        return self.field_objects