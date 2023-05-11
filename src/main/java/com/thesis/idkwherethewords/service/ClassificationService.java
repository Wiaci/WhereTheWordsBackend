package com.thesis.idkwherethewords.service;

import com.aliasi.classify.Classification;
import com.aliasi.classify.Classified;
import com.aliasi.classify.DynamicLMClassifier;
import com.aliasi.classify.JointClassification;
import com.aliasi.lm.NGramProcessLM;
import com.thesis.idkwherethewords.entity.Text;
import com.thesis.idkwherethewords.entity.Topic;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class ClassificationService {

    private final TextService textService;
    private final TopicService topicService;
    private DynamicLMClassifier<NGramProcessLM> classifier;
    private Set<String> topics;

    @PostConstruct
    public void trainModel() {
        System.out.println("trainModel");
        List<Text> texts = textService.findAll();
        topics = new HashSet<>();
        texts.forEach(text -> text.getTopics()
                .forEach(topic -> topics.add(topic.getName())));

        classifier = DynamicLMClassifier.createNGramProcess(
                        topics.toArray(new String[0]), 8);
        for (Text text : texts) {
            for (Topic topic : text.getTopics()) {
                System.out.println("Text " + text.getId() + " topic " + topic.getName());
                String t = text.getRawText();
                Classification classification = new Classification(topic.getName());
                Classified<CharSequence> classified = new Classified<>(t, classification);
                classifier.handle(classified);
            }
        }

    }

    public List<Topic> getTextTopics(String text) {
        JointClassification jc = classifier.classify(text);
        topics.forEach(s -> System.out.println(s + " " + jc.conditionalProbability(s)));
        List<String> selectedTopics = topics.stream()
                .filter(topic -> jc.conditionalProbability(topic) > 0)
                .sorted((s1, s2) -> Double.compare(jc.conditionalProbability(s2), jc.conditionalProbability(s1)))
                .limit(3)
                .collect(Collectors.toList());
        return topicService.getTopicListByNames(selectedTopics);
    }
}
