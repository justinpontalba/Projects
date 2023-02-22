# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 12:42:35 2019

@author: Justi
"""

#class Dog:
#    
#    # Class Attribute
#    species = "mammal"
#    
#    
#    # Initializer / Instance Attributes
#    def __init__(self, name, age):
#        self.name = name
#        self.age = age
#        
#    def bark(self):
#        print("bark bark!")
#        
#    def doginfo(self):
#        print(self.name + " is " + str(self.age) + " year(s) old.")
#        
#    def birthday(self):
#        self.age +=1
#        
#    def setBuddy(self, buddy):
#        self.buddy = buddy
#        buddy.buddy = self
#        
#        
#
#ozzy = Dog("Ozzy", 2)
#skippy = Dog("Skippy", 12)
#filou = Dog("Filou", 8)
#
#
#ozzy.doginfo()
#skippy.doginfo()
#filou.doginfo()
#
#ozzy.birthday()
#print(ozzy.age)
#
#print(ozzy.buddy.name)
#print(ozzy.buddy.age)
#print(filou.buddy.name)
#print(filou.buddy.age)

class Dog:
    
    # Class attribute
    species = 'mammal'
    
    # Initializer / Instance attributes
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    # instance method
    def description(self):
        return "{} is {} year(s) old".format(self.name, self.age)
    
    # instance method
    def speak(self, sound):
        return "{} says {}".format(self.name, sound)
    
class RusselTerrier(Dog):
    
    def run(self, speed):
        return "{} runs {}".format(self.name, speed)
    
class Bulldog(Dog):
    
    def run(self, speed):
        return "{} runs {}".format(self.name, speed)


# Is Julie an instance of Dog()?
julie = Dog("Julie", 100)
print(isinstance(julie, Dog))


# Is jim an instance of Dog()?
jim = Bulldog("Jim", 12)
print(isinstance(jim,Dog))
print(jim.description())
print(jim.run('slowly'))

# Is johnny walker an instance of Bulldog()?
johnnywalker = RusselTerrier("Johnny Walker", 4)
print(isinstance(johnnywalker, Bulldog))
print(isinstance(johnnywalker, RusselTerrier))
print(isinstance(johnnywalker, Dog))

class Cat:
    species = "mammal"
    
class someBreed(Cat):
    pass

class someotherbreed(Cat):
    species = 'reptile'
    
frank = someBreed()
frank.species

beans = someotherbreed()
beans.species

class Pet:
    
    dogs = []
    
    def __init__(self, dogs):
        
        self.dogs = dogs
        


dog_list = [Dog("Tom", 6), Dog("Fletcher", 7), Dog("Larry", 9)]

pets = Pet(dog_list)

for dog in pets.dogs:
    print("{} is {}.".format(dog.name, dog.age))



