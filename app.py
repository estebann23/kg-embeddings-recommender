import logging
import os
from json import dumps
from textwrap import dedent
from typing import cast

import neo4j
from flask import Flask, Response, request
from neo4j import GraphDatabase, basic_auth
from typing_extensions import LiteralString
