.. about.rst: 

About Tile
==========

Tile is a :abbr:`Domain-Specific Language (DSL)` (DSL) that can 
be used to describe tensor-based operations in a clean and 
simple format.  

.. TODO add more spec detail here. 

About Stripe
------------

``Stripe`` is a polyhedral :abbr:`Intermediate Representation (IR)`, 
or *IR*, that is highly amenable to optimization, enabling things like   

* Arbitrary tensorization

* Affine vertical fusion

* Arbitrarily complex memory hierarchry

* Heterogenous compute topologies

* Detailed performance and cost estimation

These and other low-level features also make it easier to write code or
co-design for software-hardware conceptual interfaces, programs, etc. 
