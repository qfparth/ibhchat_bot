from django.db import models


# Create your models here.
class ChatApp(models.Model):
    question = models.CharField(max_length=500)
    answer = models.TextField()
    is_answered = models.BooleanField(default=False)

    def __str__(self):
        return self.question
    

class Category(models.Model):
    name= models.CharField(max_length=200)

    class Meta:
        db_table = "categories"
        managed = False
    def __str__(self):
        
        return self.name
    

class City(models.Model):
    city = models.CharField(max_length=200,default=True)
    
    class Meta:
        db_table="cities"
    
class User(models.Model):
    business_name = models.CharField(max_length=200)
    phone = models.CharField(max_length=20)
    address = models.TextField()
    city = models.ForeignKey(City,on_delete=models.CASCADE,db_column="city")
    
    category = models.ForeignKey(Category,
        on_delete=models.CASCADE,
        db_column="category_id"
    )

    class Meta:
        db_table = "users"
        managed = False

    
class Company(models.Model):
    company_name = models.CharField(max_length=200)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    city = models.ForeignKey(City,on_delete=models.CASCADE,db_column="city"   # IMPORTANT
)
    description = models.TextField()

    def __str__(self):
        return self.company_name
    
    class Meta:
        db_table = "company"   # તમારા actual table નું નામ
        managed = False
    
class Service(models.Model):
    service_title = models.CharField(max_length=199)
    description = models.TextField(null=True, blank=True)
    keywords = models.TextField(null=True, blank=True)

    class Meta:
        db_table = "services"
        managed = False

    def __str__(self):
        return self.service_title
    

class ProfileDailyVisit(models.Model):
    profile_id = models.IntegerField()
    visit_date = models.DateField()
    visits = models.IntegerField(default=0)

    class Meta:
        db_table = "profile_daily_visits"
        managed = False

    