from multiprocessing import Value

class Counter:
    """
    A process safe counter.
    """
    def __init__(self):
        self._count = Value("i", 0)

    def increment(self, increment_value: int=1) -> None:
        """
        Increment counter by increment_value.
        """
        with self._count.get_lock():
            self._count.value += increment_value

    def get(self) -> int:
        """
        Get counter value.
        """
        with self._count.get_lock():
            return self._count.value

    def set(self, value: int) -> None:
        """
        Set counter value.
        """
        with self._count.get_lock():
            self._count.value = value
