import java.util.Scanner;

public class GoblinGame {
    private static Scanner scanner = new Scanner(System.in);
    private static int health = 100;
    private static int gold = 50;
    private static boolean hasSword = false;
    private static boolean hasAmulet = false;
    
    public static void main(String[] args) {
        System.out.println("====================================");
        System.out.println("    ДОБРО ПОЖАЛОВАТЬ В ЛАВКУ ГРОГА    ");
        System.out.println("====================================");
        System.out.println("Ты - путник, зашедший в темную пещеру.");
        System.out.println("Перед тобой появляется зеленый гоблин по имени Грог...\n");
        
        startGame();
    }
    
    private static void startGame() {
        System.out.println("Грог: 'Приветствую, путник! Хочешь заработать легких денег?'");
        System.out.println("Грог: 'У меня есть для тебя особенное предложение...'\n");
        
        firstChoice();
    }
    
    private static void firstChoice() {
        System.out.println("Грог протягивает тебе странный амулет, который светится красным.");
        System.out.println("Он предлагает обменять его на весь твой золотой запас.\n");
        System.out.println("Что выберешь?");
        System.out.println("1. Согласиться на сделку (отдать 50 золота за амулет)");
        System.out.println("2. Отказаться и спросить про другое предложение");
        System.out.println("3. Попытаться убежать из пещеры");
        System.out.println("4. Ударить гоблина");
        
        int choice = getUserChoice(1, 4);
        
        switch(choice) {
            case 1:
                buyAmulet();
                break;
            case 2:
                askOtherOffer();
                break;
            case 3:
                tryEscape();
                break;
            case 4:
                attackGoblin();
                break;
        }
    }
    
    private static void buyAmulet() {
        gold = 0;
        hasAmulet = true;
        
        System.out.println("\nТы отдаешь все золото и надеваешь амулет.");
        System.out.println("Грог злобно ухмыляется: 'Отличный выбор! Теперь ты мой...'");
        System.out.println("Амулет начинает сжиматься на твоей шее!");
        
        System.out.println("\n================= КОНЦОВКА 1 =================");
        System.out.println("Амулет был проклят! Ты превращаешься в статую.");
        System.out.println("Грог пополняет свою коллекцию 'доверчивых идиотов'.");
        System.out.println("==============================================");
        playAgain();
    }
    
    private static void askOtherOffer() {
        System.out.println("\nГрог: 'Хм... Тогда может купишь мой волшебный меч за 30 золота? Он никогда не тупится!'\n");
        System.out.println("Что выберешь?");
        System.out.println("1. Купить меч (потратить 30 золота)");
        System.out.println("2. Торговаться (предложить 15 золота)");
        System.out.println("3. Отказаться и уйти");
        System.out.println("4. Обвинить гоблина в мошенничестве");
        
        int choice = getUserChoice(1, 4);
        
        switch(choice) {
            case 1:
                buySword();
                break;
            case 2:
                bargain();
                break;
            case 3:
                leave();
                break;
            case 4:
                accuseGoblin();
                break;
        }
    }
    
    private static void buySword() {
        if (gold >= 30) {
            gold -= 30;
            hasSword = true;
            
            System.out.println("\nТы покупаешь меч и выходишь из пещеры...");
            System.out.println("На выходе на тебя нападает огромный тролль!");
            System.out.println("Ты достаешь 'волшебный' меч...");
            System.out.println("Меч ломается при первом же ударе! Это была подделка!");
            
            System.out.println("\n================= КОНЦОВКА 2 =================");
            System.out.println("Тролль разрывает тебя на части.");
            System.out.println("Грог наблюдает за этим и смеется: 'Еще один лопух!'");
            System.out.println("==============================================");
        } else {
            System.out.println("\nУ тебя нет столько золота!");
            System.out.println("Грог разочарованно вздыхает и выгоняет тебя из пещеры пинками.");
            System.out.println("\n================= КОНЦОВКА 3 =================");
            System.out.println("Ты остался без денег и с больным местом, по которому тебя пнули.");
            System.out.println("==============================================");
        }
        playAgain();
    }
    
    private static void bargain() {
        if (gold >= 15) {
            gold -= 15;
            System.out.println("\nГрог: 'Ну ладно, ладно... 15 так 15. Держи свой меч.'");
            System.out.println("Ты получаешь меч, но он оказывается ржавым и тупым.");
            System.out.println("Грог довольно потирает руки: 'Еще один лох...'");
            
            System.out.println("\n================= КОНЦОВКА 4 =================");
            System.out.println("Ты стал обладателем бесполезного меча и потратил половину золота.");
            System.out.println("Грог смеется тебе вслед, когда ты уходишь.");
            System.out.println("==============================================");
        } else {
            System.out.println("\nГрог: 'У тебя даже 15 золота нет? Проваливай, нищеброд!'");
            System.out.println("Грог кидает в тебя камень.");
            health -= 30;
            System.out.println("Ты теряешь 30 здоровья! Осталось: " + health);
            
            if (health <= 0) {
                System.out.println("\n================= КОНЦОВКА 5 =================");
                System.out.println("Камень попал прямо в голову... Ты умираешь.");
                System.out.println("Грог забирает твое тело и делает из него чучело.");
                System.out.println("==============================================");
            } else {
                System.out.println("\nТы убегаешь из пещеры с остатками здоровья.");
                System.out.println("\n================= КОНЦОВКА 5b =================");
                System.out.println("Ты выжил, но запомнишь этого гоблина надолго!");
                System.out.println("==============================================");
            }
        }
        playAgain();
    }
    
    private static void tryEscape() {
        System.out.println("\nТы пытаешься убежать, но Грог бросает вслед сеть!");
        System.out.println("Грог: 'Куда побежал, ужин? Ха-ха-ха!'");
        
        System.out.println("\n================= КОНЦОВКА 6 =================");
        System.out.println("Грог связывает тебя и жарит на ужин.");
        System.out.println("==============================================");
        playAgain();
    }
    
    private static void attackGoblin() {
        System.out.println("\nТы бьешь Грога кулаком!");
        System.out.println("Грог: 'Ай! Больно! Ща получишь!'");
        System.out.println("Грог зовет своих братьев, и они набрасываются на тебя.");
        
        System.out.println("\n================= КОНЦОВКА 7 =================");
        System.out.println("Ты проиграл битву 5 гоблинам.");
        System.out.println("Теперь ты будешь работать на них в шахте вечно.");
        System.out.println("==============================================");
        playAgain();
    }
    
    private static void leave() {
        System.out.println("\nТы решаешь уйти из пещеры...");
        System.out.println("Грог: 'Ладно, иди. Но запомни - здесь тебе всегда рады!'");
        System.out.println("Он зловеще смеется тебе вслед.");
        
        if (hasSword) {
            System.out.println("\n================= КОНЦОВКА 8 =================");
            System.out.println("Ты уходишь с бесполезным мечом, но хотя бы живой.");
            System.out.println("Грог все равно тебя надул, но ты остался в выигрыше - ты жив!");
            System.out.println("==============================================");
        } else if (hasAmulet) {
            System.out.println("\n================= КОНЦОВКА 9 =================");
            System.out.println("Ты носишь проклятый амулет. Через неделю ты превратишься в гоблина.");
            System.out.println("Теперь ты сам будешь обманывать путников!");
            System.out.println("==============================================");
        } else {
            System.out.println("\n================= КОНЦОВКА 10 =================");
            System.out.println("Ты уходишь с золотом в кармане и неповрежденной гордостью.");
            System.out.println("Грог: 'Приходи еще, когда захочешь приключений!'");
            System.out.println("==============================================");
        }
        playAgain();
    }
    
    private static void accuseGoblin() {
        System.out.println("\nТы: 'Ты мошенник! Твой меч - дешевая подделка!'");
        System.out.println("Грог смеется: 'Конечно подделка! Ты только что догадался?'");
        System.out.println("Грог восхищен твоей проницательностью и дарит тебе золотую монету.");
        gold++;
        
        System.out.println("\n================= КОНЦОВКА 11 =================");
        System.out.println("Ты получил 1 золотой за смекалку!");
        System.out.println("Грог: 'Ты первый кто меня раскусил! Уважаю!'");
        System.out.println("Вы становитесь друзьями и вместе обманываете других путников.");
        System.out.println("==============================================");
        playAgain();
    }
    
    private static int getUserChoice(int min, int max) {
        int choice = 0;
        boolean validInput = false;
        
        while (!validInput) {
            System.out.print("Твой выбор (" + min + "-" + max + "): ");
            try {
                choice = Integer.parseInt(scanner.nextLine());
                if (choice >= min && choice <= max) {
                    validInput = true;
                } else {
                    System.out.println("Пожалуйста, введи число от " + min + " до " + max);
                }
            } catch (NumberFormatException e) {
                System.out.println("Пожалуйста, введи число!");
            }
        }
        return choice;
    }
    
    private static void playAgain() {
        System.out.println("\nХочешь попробовать другую концовку?");
        System.out.println("1. Начать заново");
        System.out.println("2. Выйти из игры");
        
        int choice = getUserChoice(1, 2);
        
        if (choice == 1) {
            // Сброс параметров
            health = 100;
            gold = 50;
            hasSword = false;
            hasAmulet = false;
            
            System.out.println("\n\n\n============ НОВАЯ ИГРА ============\n");
            startGame();
        } else {
            System.out.println("\nСпасибо за игру! Приходи еще к Грогу!");
            scanner.close();
            System.exit(0);
        }
    }
}