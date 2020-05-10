class Car:
    def __init__(self, car_brand):
       self.car_brand = car_brand
    @property
    def car_brand(self):
        return self._car_brand
    #setter function
    @car_brand.setter
    def car_brand(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._car_brand = value
    
    #deleter function
    @car_brand.deleter
    def car_brand(self):
        raise AttributeError("Can't delete attribute")
       
       
car1 = Car('Tesla')
print("Car brand:", car1.car_brand)
del car1.car_brand
print("Car brand:", car1.car_brand)

car2 = Car(500)
print("Car brand:", car2.car_brand)
