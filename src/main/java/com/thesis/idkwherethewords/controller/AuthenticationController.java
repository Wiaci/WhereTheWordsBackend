package com.thesis.idkwherethewords.controller;

import com.thesis.idkwherethewords.dto.UserDetailsDto;
import com.thesis.idkwherethewords.dto.UserDto;
import com.thesis.idkwherethewords.entity.User;
import com.thesis.idkwherethewords.entity.Word;
import com.thesis.idkwherethewords.entity.Wordlist;
import com.thesis.idkwherethewords.entity.enumerated.Role;
import com.thesis.idkwherethewords.repository.WordListRepo;
import com.thesis.idkwherethewords.service.TopicService;
import com.thesis.idkwherethewords.service.UserService;
import com.thesis.idkwherethewords.service.WordListService;
import com.thesis.idkwherethewords.service.WordService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequiredArgsConstructor
public class AuthenticationController {

    private final UserService userService;
    private final WordService wordService;
    private final TopicService topicService;
    private final WordListService wordListService;

    @PostMapping("/login")
    public ResponseEntity<UserDto> login(@RequestBody UserDetailsDto userDetails) {
        System.out.println(userDetails.getUsername() + " " + userDetails.getPassword());
        UserDto userDto = userService.authenticateUser(userDetails);
        ResponseEntity<UserDto> entity;
        if (userDto.getId() > 0) {
            entity = new ResponseEntity<>(userDto,  HttpStatus.OK);
        } else {
            entity = new ResponseEntity<>(new UserDto(), HttpStatus.UNAUTHORIZED);
        }
        return entity;
    }

    @PostMapping("/checkUsername")
    public ResponseEntity<Boolean> checkUsername(@RequestBody String username) {
        return new ResponseEntity<>(userService.userExistsByName(username),HttpStatus.OK);
    }

    @PostMapping("/newUser")
    public ResponseEntity<UserDto> newUser(@RequestBody UserDto userDto) {
        List<Word> words = wordService.getSimplestUnknownWords(userDto.getWords());
        String username = userDto.getUsername().split(":")[0];
        String password = userDto.getUsername().split(":")[1];
        User user = new User(username, password, Role.USER,
                topicService.getTopicList(userDto.getTopics()));
        UserDto userDto1 = userService.addUser(user);

        User actUser = userService.getUserByName(userDto1.getUsername());
        wordListService.saveWordsForUser(actUser, words);
        return new ResponseEntity<>(userService.getUserDto(username, password), HttpStatus.OK);
    }
}
