from db import db_connection,db_operator
from nltk.ccg import chart, lexicon
from nltk.ccg.chart import CCGChartParser, ApplicationRuleSet, CompositionRuleSet
from nltk.ccg.chart import SubstitutionRuleSet, TypeRaiseRuleSet, printCCGDerivation

class grammar:
    @staticmethod
    def do():
        db = db_connection()
        selector = db_operator(db)
        db.connect()
        query = 'Select * from reviews limit 100'
        sel = selector.executeSelection(query=query)
        db.disconnect()
        lex = lexicon.parseLexicon('''
        :- S, NP, N, VP
        Det :: NP/N
        Pro :: NP
        Modal :: S\\NP/VP
        TV :: VP/NP
        DTV :: TV/NP
        the => Det
        that => Det
        that => NP
        I => Pro
        you => Pro
        we => Pro
        chef => N
        cake => N
        children => N
        dough => N
        will => Modal
        should => Modal
        might => Modal
        must => Modal
        and => var\\.,var/.,var
        to => VP[to]/VP
        without => (VP\\VP)/VP[ing]
        be => TV
        cook => TV
        eat => TV
        cooking => VP[ing]/NP
        give => DTV
        is => (S\\NP)/NP
        prefer => (S\\NP)/NP
        which => (N\\N)/(S/NP)
        persuade => (VP/VP[to])/NP
        ''')
        parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)
        for parse in parser.parse("you prefer that cake".split()):  # doctest: +SKIP
            print(chart.printCCGDerivation(parse))
        for parse in parser.parse("that is the cake which you prefer".split()):  # doctest: +SKIP
            print(chart.printCCGDerivation(parse))
        sent = "that is the dough which you will eat without cooking".split()
        nosub_parser = chart.CCGChartParser(lex, chart.ApplicationRuleSet +chart.CompositionRuleSet + chart.TypeRaiseRuleSet)
        for parse in nosub_parser.parse(sent):
           chart.printCCGDerivation(parse)
        for parse in parser.parse(sent):  # doctest: +SKIP
            print(chart.printCCGDerivation(parse))
        test1_lex = '''
        :- S,N,NP,VP
        I => NP
        you => NP
        will => S\\NP/VP
        cook => VP/NP
        which => (N\\N)/(S/NP)
        and => var\\.,var/.,var
        might => S\\NP/VP
        eat => VP/NP
        the => NP/N
        mushrooms => N
        parsnips => N'''
        test2_lex = ''':- N, S, NP, VP
        articles => N
        the => NP/N
        and => var\\.,var/.,var
        which => (N\\N)/(S/NP)
        I => NP
        anyone => NP
        will => (S/VP)\\NP
        file => VP/NP
        without => (VP\\VP)/VP[ing]
        forget => VP/NP
        reading => VP[ing]/NP
        '''
        lex = lexicon.parseLexicon(test1_lex)
        parser = CCGChartParser(lex, ApplicationRuleSet + CompositionRuleSet + SubstitutionRuleSet)
        #Tests handling of conjunctions
        for parse in parser.parse("I will cook and might eat the mushrooms and parsnips".split()):#Tests handling of conjunctions.
            # Note that while the two derivations are different, they are semantically equivalent.
            print(printCCGDerivation(parse)) # doctest: +NORMALIZE_WHITESPACE +SKIP

        #Tests handling subject extraction. Interesting to point that the two parses are clearly semantically different.
        lex = lexicon.parseLexicon(test2_lex)
        parser = CCGChartParser(lex, ApplicationRuleSet + CompositionRuleSet + SubstitutionRuleSet)
        for parse in parser.parse("articles which I will file and forget without reading".split()):
            printCCGDerivation(parse)  # doctest: +NORMALIZE_WHITESPACE +SKIP

        lex = lexicon.parseLexicon(u'''
        :- S, N, NP, PP
        
        AdjI :: N\\N
        AdjD :: N/N
        AdvD :: S/S
        AdvI :: S\\S
        Det :: NP/N
        PrepNPCompl :: PP/NP
        PrepNAdjN :: S\\S/N
        PrepNAdjNP :: S\\S/NP
         VPNP :: S\\NP/NP
        VPPP :: S\\NP/PP
        VPser :: S\\NP/AdjI
        
        auto => N
        bebidas => N
        cine => N
        ley => N
        libro => N
        ministro => N
        panadería => N
        presidente => N
        super => N
        
        el => Det
        la => Det
        las => Det
        un => Det
        
        Ana => NP
        Pablo => NP
        
        y => var\\.,var/.,var
        
        pero => (S/NP)\\(S/NP)/(S/NP)
        
        anunció => VPNP
        compró => VPNP
        cree => S\\NP/S[dep]
        desmintió => VPNP
        lee => VPNP
        fueron => VPPP
        
        es => VPser
        
        interesante => AdjD
        interesante => AdjI
        nueva => AdjD
        nueva => AdjI
        
        a => PrepNPCompl
        en => PrepNAdjN
        en => PrepNAdjNP
        
        ayer => AdvI
        
        que => (NP\\NP)/(S/NP)
        que => S[dep]/S
        ''')
        parser = chart.CCGChartParser(lex, chart.DefaultRuleSet)
        for parse in parser.parse(u"el ministro anunció pero el presidente desmintió la nueva ley".split()):
            printCCGDerivation(parse)
        parsers = parser.parse(u"el ministro anunció pero el presidente desmintió la nueva ley".split())
        return parsers