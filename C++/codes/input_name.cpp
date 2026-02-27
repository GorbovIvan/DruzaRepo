#include <iostream>
int main()
{
    int age;
    std::string name = "a";
    std::cout << "Input your name: ";
    std::cin >> name;
    std::cout << "\nInput your age: ";
    std::cin >> age;
    std::cout << "\nHello, " << name << "! Your age is " << age << "!";
    return 0;
}
