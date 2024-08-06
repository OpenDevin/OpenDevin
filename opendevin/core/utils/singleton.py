import dataclasses


class Singleton(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        else:
            # allow updates, just update existing instance
            # perhaps not the most orthodox way to do it, though it simplifies client code
            # useful for pre-defined groups of settings
            instance = cls._instances[cls]
            for key, value in kwargs.items():
                setattr(instance, key, value)
        return cls._instances[cls]

    @classmethod
    def reset(cls):
        # used by pytest to reset the state of the singleton instances
        for instance_type, instance in cls._instances.items():
            for field_info in dataclasses.fields(instance_type):
                if dataclasses.is_dataclass(field_info.type):
                    setattr(instance, field_info.name, field_info.type())
                elif field_info.default_factory is not dataclasses.MISSING:
                    setattr(instance, field_info.name, field_info.default_factory())
                else:
                    setattr(instance, field_info.name, field_info.default)
